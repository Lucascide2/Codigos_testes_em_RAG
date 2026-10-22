from __future__ import annotations

import asyncio
import inspect
import math
import os
from dotenv import load_dotenv
from typing import TYPE_CHECKING, Any

from google import genai
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics.collections import ContextRelevance, ContextRecall, AnswerRelevancy, Faithfulness, AnswerAccuracy
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ragas.llms import InstructorBaseRagasLLM

load_dotenv()


class EvaluateResponse(BaseModel):
    metric: str
    value: float = Field(..., description="Valor da métrica, de 0 a 1")
    threshold: float
    passed: bool = Field(..., description="True se value >= threshold")


async def context_relevance(
    llm: InstructorBaseRagasLLM,
    query: str, 
    retrieved_contexts: list[str]
 ):
    """
    Retorna o resultado da métrica de relevância do contexto, que mede o quanto
    os contextos concatenados estão relacionados com a consulta do usuário, 
    e que é gerada por uma LLM.

    Args:
        llm (InstructorBaseRagasLLM): Modelo de linguagem utilizado para avaliar
            a relevância dos contextos.
        query (str): Consulta original do usuário.
        retrieved_context (list[str]): Lista de contextos recuperados que serão
            avaliados quanto à relevância em relação à consulta.

    Returns:
        float: Resultado da avaliação da relevância do contexto, que pode variar de 0 a 1.
    """

    metric = ContextRelevance(llm=llm)
    
    result = await metric.ascore(
       user_input=query,
       retrieved_contexts=retrieved_contexts 
    )

    return result.value


async def context_recall(
    llm: InstructorBaseRagasLLM,
    query: str, 
    retrieved_contexts: list[str],
    reference: str
 ):
    """
    Retorna o resultado da métrica Recall de contexto, que mede o quanto
    os contextos em conjunto estão relacionados com as colocações buscadas pela
    consulta do usuário e presentes na resposta de referência.

    Args:
        llm (InstructorBaseRagasLLM): Modelo de linguagem utilizado para avaliar
            a capacidade de recuperação dos contextos corretos pelo RAG.
        query (str): Consulta original do usuário.
        retrieved_context (list[str]): Lista de contextos recuperados que serão
            avaliados quanto à relevância em relação à consulta e à resposta de referência.
        reference (str): Resposta de Referência, da qual serão retiradas as colocações.

    Returns:
        float: Resultado da avaliação do Recall de contexto, que pode variar de 0 a 1.
    """

    metric = ContextRecall(llm=llm)
    
    result = await metric.ascore(
       user_input=query,
       retrieved_contexts=retrieved_contexts,
       reference=reference 
    )

    return result.value

async def answer_relevancy(
    llm: InstructorBaseRagasLLM,
    embeddings: GoogleEmbeddings,
    query: str,
    response: str     
):
    """
    Retorna o resultado da métrica de Relevância de resposta, que mede o quanto
    o quão relevante é a resposta gerada em relação à consulta realizada pelo usuário.

    Args:
        llm (InstructorBaseRagasLLM): Modelo de linguagem utilizado para avaliar
            a relevância da resposta gerada.
        query (str): Consulta original do usuário.
        response (str): Resposta gerada para responder à consulta do usuário.

    Returns:
        float: Resultado da avaliação da Relevância de resposta, que pode variar de 0 a 1.
    """

    metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
    
    result = await metric.ascore(
       user_input=query,
       response=response 
    )

    return result.value

async def faithfulness(
    llm: InstructorBaseRagasLLM,
    query: str, 
    response: str,
    retrieved_contexts: list[str],
 ):
    """
    Retorna o resultado da métrica de fidelidade, que mede o quanto
    a resposta gerada tem correspondências com o contexto de referência.

    Args:
        llm (InstructorBaseRagasLLM): Modelo de linguagem utilizado para avaliar
            a capacidade do modelo de entregar respostas fiéis ao contexto recuperado.
        query (str): Consulta original do usuário.
        response (str): Resposta Gerada, da qual serão retiradas as colocações.

    Returns:
        float: Resultado da avaliação da fidelidade da resposta ao
        contexto, que pode variar de 0 a 1.
    """

    metric = Faithfulness(llm=llm)
    
    result = await metric.ascore(
       user_input=query,
       response=response,
       retrieved_contexts=retrieved_contexts,
    )

    return result.value

async def answer_accuracy(
    llm: InstructorBaseRagasLLM,
    query: str,
    response: str,
    reference: str 
):
    """
    Retorna o resultado da métrica de corretude da resposta, que mede o quanto
    a resposta gerada se alinha com a resposta de referência.

    Args:
        llm (InstructorBaseRagasLLM): Modelo de linguagem utilizado para avaliar
            a corretude da resposta.
        query (str): Consulta original do usuário.
        response (str): Resposta gerada para responder à consulta do usuário.
        reference (str): Resposta de Referência, da qual serão retiradas as colocações.

    Returns:
        float: Resultado da avaliação da corretude da resposta, que pode variar de 0 a 1.
    """

    metric = AnswerAccuracy(llm=llm)
    
    result = await metric.ascore(
       user_input=query,
       response=response,
       reference=reference
    )

    return result.value


METRIC_STR_TO_METRIC_FUNC_MAP = {
    "context_relevance": context_relevance,
    "context_recall": context_recall,
    "answer_relevance": answer_relevancy,
    "faithfulness": faithfulness,
    "answer_accuracy": answer_accuracy
}

INJECTED_PARAMS = ("llm", "embeddings")

# --------------------------------------------------------------------------- #
# Erros de domínio
# --------------------------------------------------------------------------- #
class RagTestError(Exception):
    """Classe base para os erros previstos durante a execução de `rag_test`."""
 
 
class InvalidMetricError(RagTestError):
    """Indica que a string da métrica não existe no mapa de métricas."""
 
 
class MissingFieldsError(RagTestError):
    """Indica que faltam campos obrigatórios (ou estão vazios) na interação."""
 
 
class MetricComputationError(RagTestError):
    """Indica falha ao calcular a métrica (LLM, rede, parsing ou valor inválido)."""

# async def rag_test(metric_str, threshold, interaction):
#     metric_func = None

#     if metric_str in list(metric_str_to_metric_func_map.keys()):
#         metric_func = metric_str_to_metric_func_map[metric_str]
#     else:
#         raise ValueError("A string da métrica não é uma string válida")


#     func_signature = inspect.signature(metric_func)
#     valid_arguments = func_signature.parameters.keys()
#     filtered_interaction = {k:v for k, v in interaction.items() if k in valid_arguments }

#     valor_da_metrica = await metric_func(**filtered_interaction)
#     print(f"Valor da métrica {metric_str}: {valor_da_metrica}")

#     return valor_da_metrica >= threshold


async def rag_test(
    metric_str: str,
    threshold: float,
    interaction: dict[str, Any],
    llm: InstructorBaseRagasLLM,
    embeddings: GoogleEmbeddings | None = None,
) -> EvaluateResponse:
    """
    Calcula uma métrica RAGAS para uma interação e verifica se o valor obtido
    atinge o limiar informado.
 
    A função seleciona a métrica a partir de `metric_str`, valida a existência
    dos campos exigidos em `interaction` antes de acionar o LLM (evitando
    chamadas desnecessárias), injeta `llm` e `embeddings` apenas nas métricas
    que os utilizam e ignora campos de `interaction` que a métrica não usa.
 
    Args:
        metric_str (str): Nome da métrica. Valores válidos: "context_relevance",
            "context_recall", "answer_relevance", "faithfulness" e
            "answer_accuracy".
        threshold (float): Limiar de aprovação, entre 0 e 1.
        interaction (dict[str, Any]): Dados da interação a ser avaliada, com as
            chaves exigidas pela métrica (query, response, retrieved_contexts
            e/ou reference).
        llm (InstructorBaseRagasLLM): Modelo de linguagem utilizado como juiz
            na avaliação.
        embeddings (GoogleEmbeddings | None): Modelo de embeddings, necessário
            apenas para a métrica "answer_relevance".
 
    Returns:
        EvaluateResponse: Objeto com o nome da métrica, o valor calculado
        (de 0 a 1), o limiar utilizado e o indicador `passed`, que é True
        quando o valor é maior ou igual ao limiar.
 
    Raises:
        InvalidMetricError: Se `metric_str` não corresponder a uma métrica válida.
        ValueError: Se `threshold` estiver fora do intervalo de 0 a 1.
        MissingFieldsError: Se algum campo exigido pela métrica estiver
            ausente ou vazio em `interaction`.
        MetricComputationError: Se o cálculo da métrica falhar ou retornar
            um valor inválido (None ou NaN).
    """
    metric_func = METRIC_STR_TO_METRIC_FUNC_MAP.get(metric_str)
    if metric_func is None:
        raise InvalidMetricError(
            f"Métrica '{metric_str}' inválida. Opções: {sorted(METRIC_STR_TO_METRIC_FUNC_MAP)}"
        )
 
    if not 0 <= threshold <= 1:
        raise ValueError("O limiar (threshold) deve estar entre 0 e 1")
 
    params = inspect.signature(metric_func).parameters
    required = [p for p in params if p not in INJECTED_PARAMS]
 
    missing = [p for p in required if interaction.get(p) in (None, "", [])]
    if missing:
        raise MissingFieldsError(
            f"Campos ausentes ou vazios em 'interaction' para '{metric_str}': {missing}"
        )
 
    kwargs = {p: interaction[p] for p in required}
    available = {"llm": llm, "embeddings": embeddings}
    for name in INJECTED_PARAMS:
        if name in params:
            kwargs[name] = available[name]
 
    try:
        value = await metric_func(**kwargs)
    except Exception as exc:
        raise MetricComputationError(f"Erro ao calcular '{metric_str}': {exc}") from exc
 
    if value is None or math.isnan(float(value)):
        raise MetricComputationError(f"A métrica '{metric_str}' não retornou um valor válido")
 
    value = float(value)
    passed = value >= threshold
    print(f"Valor da métrica {metric_str}: {value}")
 
    return EvaluateResponse(metric=metric_str, value=value, threshold=threshold, passed=passed)

async def main():
    client = genai.Client(
        api_key = os.environ["GOOGLE_API_KEY"]
    )
    llm = llm_factory(
        model='gemini-3.5-flash-lite',
        provider='google',
        client=client
    )

    embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

    interaction = {
        "query": "Qual a cor e o tamanho dos patos dali?",
        "retrieved_contexts": [
        "Os patos são pretos e grandes",
        ],
        "response":  "Os cavalos dali são pretos e grandes",
        "reference": "Os Patos ali são pretos e grandes"
    }

    interaction['llm'] = llm
    interaction['embeddings'] = embeddings

    print(f"Passou no teste de Context Relevance: {await rag_test("context_relevance", 0.70, interaction)}")
    print(f"Passou no teste de Context Recall: {await rag_test("context_recall", 0.70, interaction)}")
    print(f"Passou no teste de Answer Relevance: {await rag_test("answer_relevance", 0.70, interaction)}")
    print(f"Passou no teste de Faithfulness: {await rag_test("faithfulness", 0.70, interaction)}")
    print(f"Passou no teste de Answer Accuracy: {await rag_test("answer_accuracy", 0.70, interaction)}")

if __name__ == "__main__":
    asyncio.run(main())