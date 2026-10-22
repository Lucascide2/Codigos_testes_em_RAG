from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv
from typing import TYPE_CHECKING

from google import genai
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics.collections import ContextRelevance, ContextPrecision
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

from ragas.dataset_schema import SingleTurnSample

if TYPE_CHECKING:
    from ragas.llms import InstructorBaseRagasLLM

load_dotenv()

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

async def teste1():
    client = genai.Client(
        api_key = os.environ["GOOGLE_API_KEY"]
    )
    llm = llm_factory(
        model='gemini-3.5-flash-lite',
        provider='google',
        client=client
    )

    retrieved_contexts = [
        "O Brasil é um país que se classifica como",
        "A Holanda é um país legal",
        "Patos são legais"
    ]

    query = "Como o Brasil se classifica em relação ao seu tamanho?"

    context_relevance_var = await context_relevance(
        llm=llm,
        query=query,
        retrieved_contexts=retrieved_contexts
    )
    print(f"Relevância do contexto: {context_relevance_var}")

async def teste2():
    client = genai.Client(
        api_key = os.environ["GOOGLE_API_KEY"]
    )
    llm = llm_factory(
        model='gemini-3.5-flash-lite',
        provider='google',
        client=client
    )

    retrieved_contexts = [
        "É inegável que os patos são chatos",
    ]

    reference = "Os Patos ali são pretos e grandes"

    query = "Qual a cor e o tamanho dos patos dali?"

    retrieved_contexts3 = [
        "Os patos ali são pretos",
    ]

    context_recall_var = await context_recall(
        llm=llm,
        query=query,
        retrieved_contexts=retrieved_contexts3,
        reference=reference
    )
    print("Resultado contextos recuperados 3:")
    print(f"\tRecall do contexto: {context_recall_var}")

async def teste3():
    client = genai.Client(
        api_key = os.environ["GOOGLE_API_KEY"]
    )
    llm = llm_factory(
        model='gemini-3.5-flash-lite',
        provider='google',
        client=client
    )

    embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

    retrieved_contexts = [
        "É inegável que os patos são chatos",
    ]

    reference = "Os Patos ali são pretos e grandes"

    query = "Qual a cor e o tamanho dos patos dali?"
    response = "Os patos dali são gregos"

    answer_relevancy_var = await answer_relevancy(
        llm=llm,
        embeddings=embeddings,
        query=query,
        response=response
    )

    print("Relevância da resposta gerada 1:")
    print(f"\tRelevância da resposta: {answer_relevancy_var}")

    query = "Qual a cor e o tamanho dos patos dali?"
    response = "Os cavalos dali são pretos e grandes"

    answer_relevancy_var = await answer_relevancy(
        llm=llm,
        embeddings=embeddings,
        query=query,
        response=response
    )
    print("Relevância da resposta gerada 2:")
    print(f"\tRelevância da resposta: {answer_relevancy_var}")

async def teste4():
    client = genai.Client(
        api_key = os.environ["GOOGLE_API_KEY"]
    )
    llm = llm_factory(
        model='gemini-3.5-flash-lite',
        provider='google',
        client=client
    )

    embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

    query = "Qual a cor e o tamanho dos patos dali?"
    response = "Os cavalos dali são pretos e grandes"

    retrieved_contexts = [
        "Os patos são pretos e grandes",
    ]

    reference = "Os Patos ali são pretos e grandes"

    query = "Qual a cor e o tamanho dos patos dali?"
    response = "Os Cavalos dali são gregos"

    faithfulness_var = await faithfulness(
        llm=llm,
        query=query,
        response=response,
        retrieved_contexts=retrieved_contexts
    )
    print("Fidelidade ao contexto 1:")
    print(f"\tValor de fidelidade: {faithfulness_var}")

    query = "Qual a cor e o tamanho dos patos dali?"
    response = "Os patos dali são pretos"

    faithfulness_var = await faithfulness(
        llm=llm,
        query=query,
        response=response,
        retrieved_contexts=retrieved_contexts
    )
    print("Fidelidade ao contexto 2:")
    print(f"\tValor de fidelidade: {faithfulness_var}")