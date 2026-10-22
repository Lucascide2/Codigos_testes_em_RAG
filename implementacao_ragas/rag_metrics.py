from __future__ import annotations

from dotenv import load_dotenv
from typing import TYPE_CHECKING, Any, Callable

from google import genai
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics.collections import ContextRelevance, ContextRecall, AnswerRelevancy, Faithfulness, AnswerAccuracy
from pydantic import BaseModel, Field
from collections import Counter
from bert_score import score

from transformers import logging

# Define o nível de log apenas para erros (oculta warnings e infos)
logging.set_verbosity_error()


class MetricsResponse(BaseModel):
    """Resposta da rota /metrics."""

    metric: str
    value: float | dict[str, float] | str
    value_type_str: str

if TYPE_CHECKING:
    from ragas.llms import InstructorBaseRagasLLM

load_dotenv()

class InvalidMetricError(Exception):
    pass


class MissingFieldsError(Exception):
    pass


class MetricComputationError(Exception):
    pass

def retrieval_accuracy(
    retrieved_contexts: list[str],
    reference_contexts: list[str]
):
    vp = 0
    for ret_ctx in retrieved_contexts:
        for ref_ctx in reference_contexts:
            if ret_ctx == ref_ctx: vp += 1
    
    total = len(retrieved_contexts) + len(reference_contexts) - vp

    return vp / total

def retrieval_precision(    
    retrieved_contexts: list[str],
    reference_contexts: list[str]
):
    vp = 0

    for ret_ctx in retrieved_contexts:
        for ref_ctx in reference_contexts:
            if ret_ctx == ref_ctx: vp += 1

    fp = len(retrieved_contexts) - vp

    return vp / (vp + fp)

def recall_at_k(
    retrieved_contexts: list[str],
    reference_contexts: list[str],
    k: int
):
    vp_in_k = 0

    for ret_ctx in retrieved_contexts[:k]:
        for ref_ctx in reference_contexts:
            if ret_ctx == ref_ctx: vp_in_k +=1

    positives = len(reference_contexts)

    return vp_in_k/positives

def mean_reciprocal_rank(
    retrieved_contexts: list[list[str]],
    reference_contexts: list[list[str]]
) -> float:
    reciprocal_ranks = []

    for ret_ctxs, ref_ctxs in zip(
        retrieved_contexts, reference_contexts
    ):
        reciprocal_rank = 0

        for rank, ret_ctx in enumerate(ret_ctxs, start=1):
            if ret_ctx in ref_ctxs:
                reciprocal_rank = 1 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    return (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks else 0.0
    )

def mean_average_precision(
    retrieved_contexts: list[list[str]],
    reference_contexts: list[list[str]]
) -> float:
    average_precisions = []

    for ret_ctxs, ref_ctxs in zip(
        retrieved_contexts, reference_contexts
    ):
        relevant = set(ref_ctxs)
        hits = 0
        precision_sum = 0

        for rank, ret_ctx in enumerate(ret_ctxs, start=1):
            if ret_ctx in relevant:
                hits += 1
                precision_sum += hits / rank

        average_precision = (
            precision_sum / len(relevant)
            if relevant else 0.0
        )

        average_precisions.append(average_precision)

    return (
        sum(average_precisions) / len(average_precisions)
        if average_precisions else 0.0
    )

def rouge_n(
    generated_text: str,
    reference_text: str,
    n: int = 1
) -> dict[str, float]:
    def get_ngrams(text: str) -> Counter:
        tokens = text.lower().split()
        return Counter(
            tuple(tokens[i:i + n])
            for i in range(len(tokens) - n + 1)
        )

    generated = get_ngrams(generated_text)
    reference = get_ngrams(reference_text)

    overlap = sum(
        (generated & reference).values()
    )

    generated_total = sum(generated.values())
    reference_total = sum(reference.values())

    precision = (
        overlap / generated_total if generated_total else 0.0
    )
    recall = (
        overlap / reference_total if reference_total else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def bertscore(
    generated_text: str,
    reference_text: str,
    model_type: str = "bert-base-multilingual-cased"
) -> dict[str, float]:
    if not generated_text.strip() or not reference_text:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    predictions = [generated_text]
    refs = [reference_text]

    P, R, F1 = score(
        predictions,
        refs,
        model_type=model_type,
        lang="pt",
        verbose=False
    )

    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }



JUDGE_PROMPT = """You are evaluating a RAG system response.

Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}

Rate the following on a scale of 1-5:
1. Faithfulness: Is the answer supported by the context?
2. Relevance: Does the answer address the question?
3. Completeness: Does the answer cover all aspects?

Provide scores and brief justification."""


async def llm_judge(
    question: str,
    context: str | list[str],
    answer: str,
    llm,
) -> str:
    """
    Avalia uma resposta RAG usando um LLM como juiz.

    Args:
        question (str): Pergunta feita ao sistema.
        context (str | list[str]): Contexto(s) recuperado(s).
        answer (str): Resposta gerada pelo sistema.
        llm: Modelo de linguagem usado como juiz.

    Returns:
        str: Texto do juiz com as notas (1-5) de fidelidade, relevância e
            completude, seguidas de uma justificativa breve.
    """
    if not isinstance(context, str):
        context = "\n\n".join(context)

    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context,
        answer=answer,
    )

    response = await llm.ainvoke(prompt)
    return getattr(response, "content", response)

METRICS: dict[str, Callable] = {
    "mrr": mean_reciprocal_rank,
    "map": mean_average_precision,
    "rouge-1": lambda generated, reference: rouge_n(
        generated, reference, n=1
    ),
    "bertscore": lambda generated, reference: bertscore(
        generated, reference
    ),
    "llm_judge": llm_judge,
}

REQUIRED_FIELDS = {
    "mrr": {"retrieved_contexts", "reference_contexts"},
    "map": {"retrieved_contexts", "reference_contexts"},
    "rouge-1": {"response", "reference"},
    "bertscore": {"response", "reference"},
    "llm_judge": {"question", "retrieved_contexts", "response"},
}


async def rag_apply_metric(
    metric_str: str,
    interaction: dict,
    llm,
    embeddings,
) -> MetricsResponse:
    metric = metric_str.lower().strip()

    if metric not in METRICS:
        raise InvalidMetricError(
            f"Métrica inválida: {metric_str}. "
            f"Métricas disponíveis: {list(METRICS)}"
        )

    required_fields = REQUIRED_FIELDS[metric_str]
    missing_fields = required_fields - interaction.keys()

    if missing_fields:
        raise MissingFieldsError(
            f"Campos obrigatórios ausentes: {sorted(missing_fields)}"
        )

    try:
        if metric == "llm_judge":
            value = await METRICS[metric](
                question=interaction["query"],
                context=interaction["retrieved_contexts"],
                answer=interaction["response"],
                llm=llm,
            )
        elif metric in {"mrr", "map"}:
            value = METRICS[metric](
                interaction["retrieved_contexts"],
                interaction["reference_contexts"],
            )
        else:
            value = METRICS[metric](
                interaction["response"],
                interaction["reference"],
            )

        if type(value) == float: value_type_str = "float"
        elif type(value) == dict: value_type_str = "dict"
        elif type(value) == str: value_type_str = "str"

        return MetricsResponse(metric=metric, value=value, value_type_str=value_type_str)

    except Exception as exc:
        raise MetricComputationError(
            f"Falha ao calcular a métrica '{metric}': {exc}"
        ) from exc