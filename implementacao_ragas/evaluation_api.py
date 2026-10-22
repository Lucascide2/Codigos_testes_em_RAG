from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from google import genai
from pydantic import BaseModel, Field
from ragas.embeddings import GoogleEmbeddings
from ragas.llms import llm_factory

from rag_funcs import rag_test, EvaluateResponse, METRIC_STR_TO_METRIC_FUNC_MAP, \
                      InvalidMetricError, MissingFieldsError, MetricComputationError
from rag_metrics import rag_apply_metric, MetricsResponse

load_dotenv()

LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "gemini-embedding-001")


class EvaluateRequest(BaseModel):
    """
    Corpo da requisição do endpoint de avaliação.
 
    Attributes:
        metric (MetricName): Nome da métrica a ser calculada.
        threshold (float): Limiar de aprovação, entre 0 e 1.
        interaction (dict[str, Any]): Campos da interação (query, response,
            retrieved_contexts, reference). Campos extras são ignorados.
    """
 
    metric: Literal[*list(METRIC_STR_TO_METRIC_FUNC_MAP.keys())]
    threshold: float = Field(..., ge=0, le=1, description="Limiar de aprovação (0 a 1)")
    interaction: dict[str, Any] = Field(
        ...,
        description=(
            "Campos da interação: query, response, retrieved_contexts, reference. "
            "Envie os exigidos pela métrica; campos extras são ignorados."
        ),
        examples=[{
            "query": "Qual a capital do Brasil?",
            "response": "Brasília.",
            "retrieved_contexts": ["Brasília é a capital federal do Brasil."],
        }],
    )

class MetricsRequest(BaseModel):
    """Corpo da requisição para a rota /metrics."""

    metric: str
    interaction: dict


# --------------------------------------------------------------------------- #
# Aplicação
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
 
    Na inicialização, cria uma única vez o cliente Gemini, o LLM juiz e o modelo
    de embeddings, armazenando-os em `app.state` para reaproveitamento entre
    requisições.
 
    Args:
        app (FastAPI): Instância da aplicação, onde os recursos são armazenados.
 
    Raises:
        RuntimeError: Se a variável de ambiente GOOGLE_API_KEY não estiver definida.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Defina GOOGLE_API_KEY no ambiente ou no arquivo .env")
 
    client = genai.Client(api_key=api_key)
    app.state.llm = llm_factory(LLM_MODEL, provider="google", client=client)
    app.state.embeddings = GoogleEmbeddings(client=client, model=EMBEDDING_MODEL)
    yield


app = FastAPI(
    title="RAG Evaluation API",
    description="Expõe métricas RAGAS (relevância de contexto, recall, fidelidade etc.).",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", tags=["infra"])
async def health():
    """
    Verifica se a API está em execução.
 
    Returns:
        dict: Status da API e os nomes dos modelos de LLM e de embeddings em uso.
    """
    return {"status": "ok", "llm_model": LLM_MODEL, "embedding_model": EMBEDDING_MODEL}

@app.post("/evaluate", response_model=EvaluateResponse, tags=["evaluation"])
async def evaluate(body: EvaluateRequest, request: Request) -> EvaluateResponse:
    """
    Avalia uma interação RAG com a métrica escolhida e compara com o limiar.
 
    Args:
        body (EvaluateRequest): Métrica, limiar e dados da interação.
        request (Request): Requisição atual, usada para acessar o LLM e os
            embeddings criados no `lifespan`.
 
    Returns:
        EvaluateResponse: Valor da métrica, limiar e indicador de aprovação.
 
    Raises:
        HTTPException: 400 se a métrica for inválida, 422 se faltarem campos
            da interação e 502 se o cálculo da métrica falhar.
    """
    try:
        return await rag_test(
            metric_str=body.metric,
            threshold=body.threshold,
            interaction=body.interaction,
            llm=request.app.state.llm,
            embeddings=request.app.state.embeddings,
        )
    except InvalidMetricError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except MissingFieldsError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except MetricComputationError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

@app.post("/metrics", response_model=MetricsResponse, tags=["evaluation"])
async def metrics(body: MetricsRequest, request: Request) -> MetricsResponse:
    """
    Calcula o valor de uma métrica para uma interação RAG, sem comparar com limiar.

    Args:
        body (MetricsRequest): Métrica escolhida e dados da interação.
        request (Request): Requisição atual, usada para acessar o LLM e os
            embeddings criados no `lifespan`.

    Returns:
        MetricsResponse: Nome da métrica (normalizado) e o valor calculado.

    Raises:
        HTTPException: 400 se a métrica for inválida, 422 se faltarem campos
            da interação e 502 se o cálculo da métrica falhar.
    """
    try:
        response = await rag_apply_metric(
            metric_str=body.metric.lower().strip(),
            interaction=body.interaction,
            llm=request.app.state.llm,
            embeddings=request.app.state.embeddings,
        )

        return response

    except InvalidMetricError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except MissingFieldsError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except MetricComputationError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

if __name__ == "__main__":
    uvicorn.run("evaluation_api:app", host="0.0.0.0", port=8000, reload=True)