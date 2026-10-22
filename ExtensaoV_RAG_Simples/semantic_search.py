"""Semantic search over the locally persisted FAISS database.

The returned chunks are normal Python strings, so they are displayed as
readable text (words), not as tokenizer IDs or model tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS

from vector_db_creation import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_DIRECTORY,
    load_vector_db,
)


def search_similar_chunks(
    prompt: str,
    vector_store: FAISS,
    number_of_chunks: int = 3,
) -> list[dict[str, Any]]:
    """Return the most similar chunks and their cosine scores.

    Each result contains the readable chunk in ``text``, its cosine
    similarity in ``score`` and the original PDF metadata in ``metadata``.
    """
    if not prompt.strip():
        raise ValueError("prompt must contain at least one non-whitespace character.")
    if number_of_chunks <= 0:
        raise ValueError("number_of_chunks must be greater than zero.")

    embeddings = vector_store.embedding_function
    query_vector = np.asarray(embeddings.embed_query(prompt), dtype=np.float32)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        raise ValueError("The prompt produced a zero embedding.")
    query_vector /= query_norm

    index = vector_store.index
    vectors = np.asarray(
        [index.reconstruct(i) for i in range(index.ntotal)],
        dtype=np.float32,
    )
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / np.maximum(vector_norms, 1e-12)
    scores = normalized_vectors @ query_vector

    top_indexes = np.argsort(-scores)[:number_of_chunks]
    results: list[dict[str, Any]] = []
    for index_position in top_indexes:
        document_id = vector_store.index_to_docstore_id[int(index_position)]
        document = vector_store.docstore.search(document_id)
        if not isinstance(document, Document):
            raise TypeError(f"Invalid document stored for FAISS id: {document_id}")
        results.append(
            {
                "text": document.page_content,
                "score": float(scores[index_position]),
                "metadata": document.metadata,
            }
        )
    return results


def print_search_results(results: list[dict[str, Any]]) -> None:
    """Print chunks as readable text instead of token IDs."""
    for position, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(f"\n--- Chunk {position} | cosine={result['score']:.4f} ---")
        print(f"Arquivo: {metadata.get('file_name', metadata.get('source'))}")
        print(f"Páginas: {metadata.get('pages', metadata.get('page', 'desconhecida'))}")
        print(result["text"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the local FAISS database using cosine similarity."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Text used to search the PDF chunks. If omitted, it is requested interactively.",
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=DEFAULT_VECTOR_DIRECTORY,
        help="Directory containing index.faiss and index.pkl.",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to return.")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    vector_store = load_vector_db(args.vector_dir, args.model)
    prompt = args.prompt or input("\nDigite o prompt para a busca semântica: ")
    results = search_similar_chunks(prompt.lower(), vector_store, args.k)
    print_search_results(results)


if __name__ == "__main__":
    main()