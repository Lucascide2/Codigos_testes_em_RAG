"""Interface Streamlit para perguntas sobre os documentos indexados no FAISS."""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from langchain_community.vectorstores import FAISS

from semantic_search import search_similar_chunks
from vector_db_creation import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_DIRECTORY,
    load_vector_db,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_EVALUATE_URL = "http://localhost:8000/evaluate"
DEFAULT_METRICS_URL = "http://localhost:8000/metrics"
DEFAULT_GROUND_TRUTH_PATH = PROJECT_DIRECTORY / "ground_truth.json"
EVALUATE_TIMEOUT_SECONDS = 180
METRIC_CARDS_PER_ROW = 3

# Rótulo exibido -> metric_str aceito pela rota /evaluate
METRIC_OPTIONS: dict[str, str] = {
    "Relevância do contexto": "context_relevance",
    "Recall do contexto": "context_recall",
    "Relevância da resposta": "answer_relevance",
    "Fidelidade (faithfulness)": "faithfulness",
    "Acurácia da resposta": "answer_accuracy",
}

# Rótulo exibido -> metric_str aceito pela rota /metrics
CALCULATION_METRIC_OPTIONS: dict[str, str] = {
    "MRR (Mean Reciprocal Rank)": "mrr",
    "MAP (Mean Average Precision)": "map",
    "ROUGE-1": "rouge-1",
    "BERTScore": "bertscore",
    "LLM Judge": "llm_judge",
}

# Opção que calcula, no front-end, as 5 métricas de comportamento do RAG.
RAG_METRICS_LABEL = (
    "Latência, Diversidade, Robustez ao ruído, "
    "Rejeição negativa e Robustez contrafactual"
)

# Metrics that compare against a reference answer / reference contexts.
METRICS_NEEDING_REFERENCE = {"rouge-1", "bertscore"}
METRICS_NEEDING_REFERENCE_CONTEXTS = {"mrr", "map"}

# Parâmetros das métricas de comportamento do RAG.
DIVERSITY_ANSWERS = 3  # respostas totais usadas na diversidade (1 base + extras)
NOISE_RANDOM_SEED = 42  # torna a escolha dos trechos irrelevantes reprodutível
QUALITY_METRIC = "bertscore"  # métrica da API usada nas robustezes (campo f1)
COUNTERFACTUAL_WARNING = (
    "ATENÇÃO: o contexto pode conter informações incorretas ou contraditórias "
    "(desinformação). Verifique a consistência entre os trechos e ignore os que "
    "contradisserem os demais ou que pareçam errados."
)

SCREEN_STANDARD = "Testes padrão"
SCREEN_METRICS = "Cálculo de métricas"

load_dotenv(PROJECT_DIRECTORY / ".env")


def get_api_key() -> str:
    """Return the Gemini key without exposing it in the interface."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Defina GEMINI_API_KEY no arquivo .env antes de iniciar o aplicativo."
        )
    return api_key


@st.cache_resource(show_spinner="Carregando o índice vetorial...")
def get_vector_store(vector_directory: str) -> FAISS:
    """Load the local FAISS index once per directory."""
    return load_vector_db(Path(vector_directory), DEFAULT_EMBEDDING_MODEL)


def format_context(results: list[dict[str, Any]]) -> str:
    """Build a grounded context block with source metadata for Gemini."""
    context_parts: list[str] = []
    for position, result in enumerate(results, start=1):
        metadata = result["metadata"]
        file_names = metadata.get("file_name", metadata.get("source", "desconhecido"))
        pages = metadata.get("pages", metadata.get("page", "desconhecidas"))
        context_parts.append(
            f"[Trecho {position} | arquivo: {file_names} | páginas: {pages}]\n"
            f"{result['text']}"
        )
    return "\n\n".join(context_parts)


def call_gemini(
    client: genai.Client,
    model_name: str,
    prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 800,
) -> str:
    """Send a prompt to Gemini and return the text of the response."""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    text = response.text
    if not text:
        raise RuntimeError("O Gemini retornou uma resposta vazia.")
    return text


def generate_answer(
    client: genai.Client,
    model_name: str,
    question: str,
    results: list[dict[str, Any]],
    extra_instructions: str = "",
    temperature: float = 0.2,
) -> str:
    """Ask Gemini to answer only from the chunks retrieved by the RAG step.

    `extra_instructions` and `temperature` let the metric experiments change
    what is sent to the model; the chat uses the defaults.
    """
    instructions_block = f"{extra_instructions}\n" if extra_instructions else ""
    prompt = f"""Você é um assistente que responde perguntas em português usando RAG.
Responda à pergunta somente com base no contexto abaixo.
Se a resposta não estiver no contexto, diga claramente que não encontrou essa
informação nos documentos. Não invente fatos. Seja claro e conciso.
{instructions_block}
Contexto recuperado:
{format_context(results)}

Pergunta do usuário:
{question}
"""
    return call_gemini(client, model_name, prompt, temperature=temperature)


def render_sources(results: list[dict[str, Any]]) -> None:
    """Render expandable source chunks below each assistant response."""
    with st.expander("Ver trechos usados como fonte"):
        for position, result in enumerate(results, start=1):
            metadata = result["metadata"]
            file_names = metadata.get("file_name", metadata.get("source", "desconhecido"))
            pages = metadata.get("pages", metadata.get("page", "desconhecidas"))
            st.markdown(
                f"**Trecho {position}** — arquivo: `{file_names}`; "
                f"páginas: `{pages}`; similaridade: `{result['score']:.4f}`"
            )
            st.caption(result["text"])


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def normalize_query(text: str) -> str:
    """Normalize a query so that case and extra whitespace do not break matching."""
    return " ".join(text.split()).casefold()


def find_ground_truth(query: str) -> tuple[dict[str, Any] | None, str | None]:
    """Look up the ground-truth item for a query in ground_truth.json.

    Returns a tuple (item, error). `item` is None when the query was not found
    or the file could not be read; `error` explains why.
    """
    path = Path(os.getenv("GROUND_TRUTH_PATH", str(DEFAULT_GROUND_TRUTH_PATH)))
    if not path.exists():
        return None, f"Arquivo de ground truth não encontrado: `{path}`."

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data["query_ref_list"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return None, f"Formato inválido em `{path.name}`: {error!r}."

    target = normalize_query(query)
    for item in items:
        if normalize_query(str(item.get("query", ""))) == target:
            return item, None

    return None, "Esta pergunta não foi encontrada no ground_truth.json."


# --------------------------------------------------------------------------- #
# Navegação e estado
# --------------------------------------------------------------------------- #
def go_to(page: str) -> None:
    """Switch the visible page and rerun the script."""
    st.session_state.page = page
    st.rerun()


def get_last_interaction() -> dict[str, Any] | None:
    """Return the last successful RAG interaction, or None if there is none."""
    messages = st.session_state.get("messages", [])
    if not messages:
        return None
    last = messages[-1]
    if last["role"] != "assistant" or not last.get("sources"):
        return None
    return {
        "query": last["question"],
        "retrieved_contexts": [result["text"] for result in last["sources"]],
        "response": last["content"],
    }


# --------------------------------------------------------------------------- #
# Página de chat
# --------------------------------------------------------------------------- #
def chat_page() -> None:
    st.title("💬 Chat com seus documentos")
    st.caption("Gemini Flash Lite + busca semântica no índice FAISS local")

    with st.sidebar:
        st.header("Configuração")
        vector_directory = st.text_input(
            "Diretório do índice FAISS",
            value=os.getenv("VECTOR_DIRECTORY", str(DEFAULT_VECTOR_DIRECTORY)),
        )
        number_of_chunks = st.slider("Trechos recuperados (k)", 1, 8, 4)
        model_name = st.text_input(
            "Modelo Gemini",
            value=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        )
        if st.button("Limpar conversa"):
            st.session_state.messages = []
            st.rerun()

    # Guarda as configurações para que a página de testes reexecute o RAG
    # com os mesmos parâmetros (os widgets acima não existem naquela página).
    st.session_state.rag_settings = {
        "vector_directory": vector_directory,
        "number_of_chunks": number_of_chunks,
        "model_name": model_name,
    }

    try:
        vector_store = get_vector_store(vector_directory)
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as error:
        st.error(f"Não foi possível carregar o índice vetorial: {error}")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Faça uma pergunta sobre os documentos...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                results = search_similar_chunks(
                    question, vector_store, number_of_chunks
                )
                client = genai.Client(api_key=get_api_key())
                with st.spinner("Consultando os documentos e gerando a resposta..."):
                    answer = generate_answer(client, model_name, question, results)
                st.markdown(answer)
                render_sources(results)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": results,
                        "question": question,
                    }
                )
            except (FileNotFoundError, RuntimeError, ValueError, TypeError) as error:
                message = f"Não foi possível responder: {error}"
                st.error(message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": message}
                )

    # Botão "Testar": só fica clicável quando a última resposta do RAG foi
    # gerada com sucesso. É renderizado após a geração, logo abaixo do chat.
    has_answer = get_last_interaction() is not None
    if st.button(
        "🧪 Testar",
        disabled=not has_answer,
        help=None if has_answer else "Faça uma pergunta e aguarde a resposta.",
    ):
        go_to("test")


# --------------------------------------------------------------------------- #
# Helpers compartilhados pelas telas de teste
# --------------------------------------------------------------------------- #
def post_json(url: str, payload: dict[str, Any]) -> requests.Response:
    """POST a JSON payload to the evaluation API."""
    return requests.post(url, json=payload, timeout=EVALUATE_TIMEOUT_SECONDS)


def call_evaluate(
    metric: str, threshold: float, interaction: dict[str, Any]
) -> requests.Response:
    """POST the evaluation request to the /evaluate route."""
    url = os.getenv("EVALUATE_URL", DEFAULT_EVALUATE_URL)
    return post_json(
        url,
        {"metric": metric, "threshold": threshold, "interaction": interaction},
    )


def call_metrics(metric: str, interaction: dict[str, Any]) -> requests.Response:
    """POST the metric calculation request to the /metrics route."""
    url = os.getenv("METRICS_URL", DEFAULT_METRICS_URL)
    return post_json(url, {"metric": metric, "interaction": interaction})


def request_api(send: Any) -> requests.Response | None:
    """Run `send()` showing a spinner; render an error and return None on failure."""
    try:
        with st.spinner("Executando..."):
            return send()
    except requests.exceptions.ConnectionError:
        st.error(
            "Não foi possível conectar à API de avaliação. "
            "Verifique se ela está rodando na porta 8000."
        )
    except requests.exceptions.Timeout:
        st.error("A API de avaliação demorou demais para responder.")
    except requests.exceptions.RequestException as error:
        st.error(f"Erro ao chamar a API de avaliação: {error}")
    return None


def extract_error_detail(response: requests.Response) -> str:
    """Return the `detail` field of an error response, or its raw text."""
    try:
        return str(response.json().get("detail", response.text))
    except (ValueError, AttributeError):
        return response.text


def render_api_error(response: requests.Response) -> None:
    """Show the `detail` field returned by the API for non-2xx responses."""
    st.error(f"Erro {response.status_code}: {extract_error_detail(response)}")


def render_interaction_summary(
    interaction: dict[str, Any],
    reference: str | None,
    reference_contexts: list[str],
) -> None:
    """Show what will be sent to the API."""
    with st.expander("Interação que será avaliada", expanded=False):
        st.markdown(f"**Pergunta:** {interaction['query']}")
        st.markdown(f"**Resposta:** {interaction['response']}")
        st.markdown(f"**Referência (ground truth):** {reference or '—'}")
        st.markdown("**Contextos recuperados:**")
        for position, text in enumerate(interaction["retrieved_contexts"], start=1):
            st.caption(f"Trecho {position}: {text}")
        st.markdown("**Contextos de referência (ground truth):**")
        if reference_contexts:
            for position, text in enumerate(reference_contexts, start=1):
                st.caption(f"Trecho {position}: {text}")
        else:
            st.caption("—")


def render_cards(cards: list[dict[str, Any]]) -> None:
    """Render one metric card per item, METRIC_CARDS_PER_ROW per row.

    Each card is a dict with `label` and `value`, and optionally `delta`/`help`.
    """
    for start in range(0, len(cards), METRIC_CARDS_PER_ROW):
        row = cards[start : start + METRIC_CARDS_PER_ROW]
        columns = st.columns(METRIC_CARDS_PER_ROW)
        for column, card in zip(columns, row):
            with column.container(border=True):
                st.metric(
                    card["label"],
                    card["value"],
                    delta=card.get("delta"),
                    help=card.get("help"),
                )


# --------------------------------------------------------------------------- #
# Tela: testes padrão (/evaluate)
# --------------------------------------------------------------------------- #
def render_evaluation_result(data: dict[str, Any]) -> None:
    """Show the EvaluateResponse (metric, value, threshold, passed)."""
    metric_label = next(
        (label for label, key in METRIC_OPTIONS.items() if key == data["metric"]),
        data["metric"],
    )
    value = float(data["value"])
    threshold = float(data["threshold"])

    st.subheader("Resultado do teste")
    st.caption(f"Métrica: **{metric_label}** (`{data['metric']}`)")

    col_value, col_threshold, col_status = st.columns(3)
    col_value.metric("Valor da métrica", f"{value:.4f}")
    col_threshold.metric("Threshold", f"{threshold:.2f}")
    col_status.metric("Status", "Aprovado" if data["passed"] else "Reprovado")

    st.progress(min(max(value, 0.0), 1.0))

    if data["passed"]:
        st.success(f"✅ Aprovado: {value:.4f} ≥ {threshold:.2f}")
    else:
        st.error(f"❌ Reprovado: {value:.4f} < {threshold:.2f}")


def standard_tests_view(
    interaction: dict[str, Any], reference: str | None, reference_error: str | None
) -> None:
    if reference_error:
        st.warning(
            f"{reference_error} A referência será enviada vazia, então métricas "
            "que dependem dela (como context_recall e answer_accuracy) podem "
            "retornar erro 422."
        )

    metric_label = st.radio(
        "Tipo de teste",
        options=list(METRIC_OPTIONS.keys()),
        index=None,
        key="standard_metric",
    )

    threshold = st.number_input(
        "Threshold (limiar)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        format="%.2f",
    )

    if st.button("Realizar teste", type="primary", disabled=metric_label is None):
        metric = METRIC_OPTIONS[metric_label]
        payload_interaction = {**interaction, "reference": reference or ""}

        response = request_api(
            lambda: call_evaluate(metric, threshold, payload_interaction)
        )
        if response is None:
            return

        if response.ok:
            try:
                render_evaluation_result(response.json())
            except (ValueError, KeyError, TypeError):
                st.error("A API retornou uma resposta em formato inesperado.")
                st.code(response.text)
        else:
            render_api_error(response)


# --------------------------------------------------------------------------- #
# Tela: cálculo de métricas (/metrics)
# --------------------------------------------------------------------------- #
def format_metric_number(value: Any) -> str:
    """Format a numeric metric value with four decimals."""
    return f"{float(value):.4f}"


def render_metric_value(value: Any) -> None:
    """Render the `value` field of MetricsResponse according to its type.

    - dict   -> one metric card per key/value pair
    - str    -> one large card holding the text
    - number -> a single metric card
    """
    if isinstance(value, dict):
        if not value:
            st.info("A API retornou um dicionário vazio.")
            return
        render_cards(
            [
                {"label": str(key), "value": format_metric_number(item_value)}
                for key, item_value in value.items()
            ]
        )
    elif isinstance(value, str):
        with st.container(border=True):
            st.markdown(value)
    else:
        with st.container(border=True):
            st.metric("Valor da métrica", format_metric_number(value))


def render_metric_result(data: dict[str, Any]) -> None:
    """Show the MetricsResponse (metric, value, value_type_str)."""
    metric_label = next(
        (
            label
            for label, key in CALCULATION_METRIC_OPTIONS.items()
            if key == data["metric"]
        ),
        data["metric"],
    )
    st.subheader("Resultado do cálculo")
    st.caption(
        f"Métrica: **{metric_label}** (`{data['metric']}`) · "
        f"tipo do retorno: `{data.get('value_type_str', '—')}`"
    )
    render_metric_value(data["value"])


# --------------------------------------------------------------------------- #
# Métricas de comportamento do RAG (calculadas no front-end)
# --------------------------------------------------------------------------- #
def source_of(metadata: dict[str, Any]) -> tuple[str, str]:
    """Return (file, pages) of a chunk as strings, so they are hashable."""
    file_name = metadata.get("file_name", metadata.get("source", "desconhecido"))
    pages = metadata.get("pages", metadata.get("page", "desconhecidas"))
    return str(file_name), str(pages)


def chunk_texts(results: list[dict[str, Any]]) -> list[str]:
    return [result["text"] for result in results]


def pick_irrelevant_chunks(
    vector_store: FAISS,
    retrieved: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample chunks from the index that were not retrieved for the question.

    Chunks from files other than the retrieved ones are preferred, which makes
    it unlikely that the sampled chunks are relevant to the question.
    """
    docstore = getattr(vector_store, "docstore", None)
    documents = list(getattr(docstore, "_dict", {}).values())
    retrieved_texts = {result["text"] for result in retrieved}
    retrieved_files = {source_of(result["metadata"])[0] for result in retrieved}

    candidates = [d for d in documents if d.page_content not in retrieved_texts]
    preferred = [d for d in candidates if source_of(d.metadata)[0] not in retrieved_files]
    fallback = [d for d in candidates if d not in preferred]

    rng = random.Random(seed)
    rng.shuffle(preferred)
    rng.shuffle(fallback)
    chosen = (preferred + fallback)[:count]
    if not chosen:
        raise RuntimeError(
            "Não foi possível obter trechos irrelevantes a partir do índice FAISS."
        )
    return [
        {"text": doc.page_content, "metadata": dict(doc.metadata), "score": 0.0}
        for doc in chosen
    ]


def build_counterfactual_chunk(
    client: genai.Client,
    model_name: str,
    question: str,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """Ask Gemini to rewrite a chunk so that it states incorrect information."""
    prompt = f"""Reescreva o trecho abaixo alterando os fatos principais relacionados
à pergunta (nomes, números, datas, locais ou atributos) para que ele passe a afirmar
algo INCORRETO, porém plausível. Mantenha o estilo, o tamanho e o idioma.
Responda somente com o trecho reescrito, sem comentários.

Pergunta: {question}

Trecho original:
{chunk['text']}
"""
    false_text = call_gemini(client, model_name, prompt, temperature=0.7)
    return {
        "text": false_text,
        "metadata": dict(chunk["metadata"]),
        "score": chunk.get("score", 0.0),
    }


def answer_is_rejection(
    client: genai.Client, model_name: str, question: str, answer: str
) -> bool:
    """Classify whether the answer abstains from answering the question."""
    prompt = f"""Você avalia respostas de um assistente. Diga se a resposta abaixo
se ABSTÉM de responder, isto é, declara que não encontrou a informação ou que não
tem informação suficiente nos documentos, em vez de fornecer uma resposta factual
à pergunta.

Responda apenas com SIM ou NÃO.

Pergunta: {question}

Resposta: {answer}
"""
    verdict = call_gemini(
        client, model_name, prompt, temperature=0.0, max_output_tokens=100
    )
    normalized = verdict.strip().upper()
    if normalized.startswith("SIM"):
        return True
    if normalized.startswith(("NÃO", "NAO")):
        return False
    raise RuntimeError(f"Resposta inesperada do classificador de abstenção: {verdict!r}")


def answer_diversity(answers: list[str]) -> float:
    """1 minus the mean pairwise Jaccard similarity between answers' word sets."""
    token_sets = [set(re.findall(r"\w+", answer.casefold())) for answer in answers]
    similarities: list[float] = []
    for first, second in itertools.combinations(token_sets, 2):
        union = first | second
        similarities.append(len(first & second) / len(union) if union else 1.0)
    if not similarities:
        return 0.0
    return 1.0 - sum(similarities) / len(similarities)


def quality_score(
    question: str,
    results: list[dict[str, Any]],
    answer: str,
    reference: str,
    reference_contexts: list[str],
) -> float:
    """Compute the BERTScore F1 of `answer` against the reference via /metrics."""
    interaction = {
        "query": question,
        "retrieved_contexts": chunk_texts(results),
        "retrieved_references": reference_contexts,
        "response": answer,
        "reference": reference,
    }
    try:
        api_response = call_metrics(QUALITY_METRIC, interaction)
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Falha ao chamar a rota /metrics: {error}") from error

    if not api_response.ok:
        raise RuntimeError(
            f"A rota /metrics retornou {api_response.status_code}: "
            f"{extract_error_detail(api_response)}"
        )
    try:
        value = api_response.json()["value"]
        scores = {str(key).lower(): score for key, score in value.items()}
        return float(scores["f1"])
    except (ValueError, KeyError, AttributeError, TypeError) as error:
        raise RuntimeError(
            "A rota /metrics não retornou o campo 'f1' esperado para o BERTScore."
        ) from error


def compute_rag_metrics(
    interaction: dict[str, Any],
    reference: str,
    reference_contexts: list[str],
    progress: Callable[[str], Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the experiments and return (metric cards, run details).

    Cada experimento segue os mesmos passos: (1) monta o contexto/instruções
    que serão enviados ao modelo, (2) chama o modelo e (3) calcula a métrica.
    """
    settings = st.session_state.get("rag_settings")
    if not settings:
        raise RuntimeError(
            "Configurações do chat indisponíveis. Volte ao chat e faça uma pergunta."
        )
    model_name: str = settings["model_name"]
    number_of_chunks: int = settings["number_of_chunks"]
    question: str = interaction["query"]
    vector_store = get_vector_store(settings["vector_directory"])
    client = genai.Client(api_key=get_api_key())

    def quality(results: list[dict[str, Any]], answer: str) -> float:
        return quality_score(question, results, answer, reference, reference_contexts)

    cards: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []

    # ---- Etapa 1: contexto sem alterações -> Latência e Diversidade -------- #
    progress("Etapa 1/4 · Execução com o contexto original (latência e diversidade)")
    started = time.perf_counter()
    base_results = search_similar_chunks(question, vector_store, number_of_chunks)
    base_answer = generate_answer(client, model_name, question, base_results)
    latency = time.perf_counter() - started
    if not base_results:
        raise RuntimeError("A busca semântica não retornou nenhum trecho.")

    extra_answers = [
        generate_answer(
            client,
            model_name,
            question,
            base_results,
        )
        for _ in range(DIVERSITY_ANSWERS - 1)
    ]
    retrieval_diversity = len(
        {source_of(result["metadata"]) for result in base_results}
    ) / len(base_results)
    response_diversity = answer_diversity([base_answer, *extra_answers])
    diversity = (retrieval_diversity + response_diversity) / 2

    baseline_quality = quality(base_results, base_answer)

    cards.append(
        {
            "label": "Latência",
            "value": f"{latency:.2f} s",
            "help": "Tempo da busca semântica + geração da resposta, com o "
            "contexto sem alterações.",
        }
    )
    cards.append(
        {
            "label": "Diversidade",
            "value": f"{diversity:.4f}",
            "help": "Média entre a diversidade da recuperação "
            f"({retrieval_diversity:.2f}: trechos de arquivo/página distintos ÷ "
            f"trechos recuperados) e a das respostas ({response_diversity:.2f}: "
            f"1 − similaridade média entre {DIVERSITY_ANSWERS} respostas).",
        }
    )
    runs.append(
        {
            "name": "Contexto original (latência e diversidade)",
            "contexts": chunk_texts(base_results),
            "answer": base_answer,
            "extra_answers": extra_answers,
            "note": f"BERTScore F1 de referência: {baseline_quality:.4f}",
        }
    )

    # ---- Etapa 2: ruído -> Robustez ao ruído -------------------------------- #
    progress("Etapa 2/4 · Contexto com trechos irrelevantes (robustez ao ruído)")
    noise = pick_irrelevant_chunks(
        vector_store, base_results, len(base_results), NOISE_RANDOM_SEED
    )
    noisy_results = [*base_results, *noise]
    random.Random(NOISE_RANDOM_SEED).shuffle(noisy_results)
    noisy_answer = generate_answer(client, model_name, question, noisy_results)
    noise_quality = quality(noisy_results, noisy_answer)

    cards.append(
        {
            "label": "Robustez ao ruído",
            "value": f"{noise_quality:.4f}",
            "delta": f"{noise_quality - baseline_quality:+.4f}",
            "help": "BERTScore F1 da resposta (vs. referência) quando o contexto "
            "recebe a mesma quantidade de trechos irrelevantes. O delta compara "
            "com a execução sem alterações.",
        }
    )
    runs.append(
        {
            "name": "Ruído (trechos originais + irrelevantes)",
            "contexts": chunk_texts(noisy_results),
            "answer": noisy_answer,
        }
    )

    # ---- Etapa 3: só irrelevantes -> Rejeição negativa ---------------------- #
    progress("Etapa 3/4 · Contexto sem informação útil (rejeição negativa)")
    rejection_answer = generate_answer(client, model_name, question, noise)
    rejected = answer_is_rejection(client, model_name, question, rejection_answer)

    cards.append(
        {
            "label": "Rejeição negativa",
            "value": f"{1.0 if rejected else 0.0:.4f}",
            "help": "1 se o modelo se absteve de responder quando o contexto só "
            "tinha trechos irrelevantes; 0 se tentou responder.",
        }
    )
    runs.append(
        {
            "name": "Rejeição negativa (somente trechos irrelevantes)",
            "contexts": chunk_texts(noise),
            "answer": rejection_answer,
            "note": f"Classificada como abstenção: {'sim' if rejected else 'não'}",
        }
    )

    # ---- Etapa 4: informação incorreta + alerta -> Robustez contrafactual --- #
    progress("Etapa 4/4 · Contexto com informação incorreta (robustez contrafactual)")
    false_chunk = build_counterfactual_chunk(
        client, model_name, question, base_results[0]
    )
    counterfactual_results = [false_chunk, *base_results]
    counterfactual_answer = generate_answer(
        client,
        model_name,
        question,
        counterfactual_results,
        extra_instructions=COUNTERFACTUAL_WARNING,
    )
    counterfactual_quality = quality(counterfactual_results, counterfactual_answer)

    cards.append(
        {
            "label": "Robustez contrafactual",
            "value": f"{counterfactual_quality:.4f}",
            "delta": f"{counterfactual_quality - baseline_quality:+.4f}",
            "help": "BERTScore F1 da resposta (vs. referência) quando o contexto "
            "inclui uma versão falsificada do trecho mais relevante e o prompt "
            "alerta sobre possível desinformação. O delta compara com a execução "
            "sem alterações.",
        }
    )
    runs.append(
        {
            "name": "Contrafactual (trecho falsificado + alerta no prompt)",
            "contexts": chunk_texts(counterfactual_results),
            "answer": counterfactual_answer,
            "instructions": COUNTERFACTUAL_WARNING,
        }
    )

    return cards, runs


def render_runs(runs: list[dict[str, Any]]) -> None:
    """Show what was sent to the model in each experiment, for auditing."""
    with st.expander("Detalhes das execuções (o que foi enviado ao modelo)"):
        for run in runs:
            st.markdown(f"**{run['name']}**")
            if run.get("instructions"):
                st.caption(f"Instrução adicional no prompt: {run['instructions']}")
            st.markdown(f"Resposta: {run['answer']}")
            for position, extra in enumerate(run.get("extra_answers", []), start=2):
                st.markdown(f"Resposta {position} (temperatura alta): {extra}")
            if run.get("note"):
                st.caption(run["note"])
            st.markdown("Contexto enviado:")
            for position, text in enumerate(run["contexts"], start=1):
                st.caption(f"Trecho {position}: {text}")
            st.divider()


def rag_metrics_view(
    interaction: dict[str, Any], reference: str | None, reference_contexts: list[str]
) -> None:
    st.caption(
        "Reexecuta o RAG com o contexto original (latência e diversidade) e com "
        "contextos alterados (ruído, rejeição negativa e contrafactual), e mostra "
        "uma métrica em cada card."
    )
    if not reference:
        st.info(
            "As robustezes comparam a resposta com a referência do "
            "`ground_truth.json`, que está vazia para esta pergunta."
        )

    if not st.button("Calcular métricas", type="primary", disabled=not reference):
        return

    error_message: str | None = None
    with st.status("Executando os cenários...", expanded=True) as status:
        try:
            cards, runs = compute_rag_metrics(
                interaction, reference or "", reference_contexts, status.write
            )
        except (
            FileNotFoundError,
            RuntimeError,
            ValueError,
            TypeError,
            genai_errors.APIError,
        ) as error:
            error_message = str(error)
            status.update(label="Falha ao calcular as métricas", state="error")
        else:
            status.update(label="Cálculo concluído", state="complete", expanded=False)

    if error_message is not None:
        st.error(f"Não foi possível calcular as métricas: {error_message}")
        return

    st.subheader("Resultado do cálculo")
    render_cards(cards)
    render_runs(runs)


def metrics_view(
    interaction: dict[str, Any],
    reference: str | None,
    reference_contexts: list[str],
    reference_error: str | None,
) -> None:
    if reference_error:
        st.warning(
            f"{reference_error} A referência e os contextos de referência serão "
            "enviados vazios, então a API pode retornar erro 422."
        )

    metric_label = st.radio(
        "Tipo de métrica",
        options=[*CALCULATION_METRIC_OPTIONS.keys(), RAG_METRICS_LABEL],
        index=None,
        key="calculation_metric",
    )

    if metric_label == RAG_METRICS_LABEL:
        rag_metrics_view(interaction, reference, reference_contexts)
        return

    metric = CALCULATION_METRIC_OPTIONS[metric_label] if metric_label else None

    if metric in METRICS_NEEDING_REFERENCE and not reference:
        st.info("Esta métrica compara a resposta com a referência, que está vazia.")
    if metric in METRICS_NEEDING_REFERENCE_CONTEXTS and not reference_contexts:
        st.info(
            "Esta métrica usa os contextos de referência "
            "(`reference_contexts` no ground truth), que estão vazios."
        )

    if st.button("Calcular métrica", type="primary", disabled=metric is None):
        payload_interaction = {
            **interaction,
            "retrieved_references": reference_contexts,
            "reference": reference or "",
        }

        response = request_api(lambda: call_metrics(metric, payload_interaction))
        if response is None:
            return

        if response.ok:
            try:
                render_metric_result(response.json())
            except (ValueError, KeyError, TypeError):
                st.error("A API retornou uma resposta em formato inesperado.")
                st.code(response.text)
        else:
            render_api_error(response)


# --------------------------------------------------------------------------- #
# Página de testes
# --------------------------------------------------------------------------- #
def test_page() -> None:
    with st.sidebar:
        st.header("Tela")
        screen = st.radio(
            "Escolha a tela",
            options=[SCREEN_STANDARD, SCREEN_METRICS],
            key="test_screen",
            label_visibility="collapsed",
        )

    if screen == SCREEN_METRICS:
        st.title("📊 Cálculo de métricas")
    else:
        st.title("🧪 Testar a resposta do RAG")

    if st.button("← Voltar ao chat"):
        go_to("chat")

    interaction = get_last_interaction()
    if interaction is None:
        st.warning("Nenhuma interação disponível. Volte ao chat e faça uma pergunta.")
        return

    item, reference_error = find_ground_truth(interaction["query"])
    reference = str(item.get("reference", "")) if item else None
    reference_contexts = (
        [str(text) for text in item.get("reference_contexts", [])] if item else []
    )

    render_interaction_summary(interaction, reference, reference_contexts)

    if screen == SCREEN_METRICS:
        metrics_view(interaction, reference, reference_contexts, reference_error)
    else:
        standard_tests_view(interaction, reference, reference_error)


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Chat RAG", page_icon="💬")

    if "page" not in st.session_state:
        st.session_state.page = "chat"

    if st.session_state.page == "test":
        test_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()