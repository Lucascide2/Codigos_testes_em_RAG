"""Create a persistent FAISS database with fixed-size cross-page chunks.

Unlike the original script, this version concatenates all extracted PDF text
before chunking. A chunk can therefore contain text from multiple pages.
Chunks have exactly 1000 characters whenever the source has at least 1000
characters. The final chunk is moved backwards as needed, which can produce
more overlap than the configured overlap instead of adding empty characters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader


DEFAULT_PDF_DIRECTORY = Path("database")
DEFAULT_VECTOR_DIRECTORY = Path("vector_db")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 400


def extract_pdf_text(
    pdf_directory: Path,
) -> tuple[str, list[tuple[int, int, str, int]]]:
    """Return continuous text and source metadata for every PDF page."""
    pdf_files = sorted(pdf_directory.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_directory}")

    parts: list[str] = []
    page_ranges: list[tuple[int, int]] = []
    current_offset = 0

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip().lower()
            if not page_text:
                continue
            if parts:
                parts.append("\n")
                current_offset += 1
            start = current_offset
            parts.append(page_text)
            current_offset += len(page_text)
            page_ranges.append((start, current_offset, str(pdf_path), page_number))

    text = "".join(parts)
    if not text:
        raise ValueError("The PDFs were found, but no text could be extracted.")
    return text, page_ranges


def create_fixed_size_chunks(
    text: str,
    page_ranges: list[tuple[int, int, str, int]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Create fixed-size chunks without using page boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
    if not text:
        return []

    if len(text) <= chunk_size:
        starts = [0]
    else:
        step = chunk_size - chunk_overlap
        starts = list(range(0, len(text) - chunk_size + 1, step))
        final_start = len(text) - chunk_size
        if starts[-1] != final_start:
            starts.append(final_start)

    documents: list[Document] = []
    for chunk_number, start in enumerate(starts, start=1):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        matching_pages = [
            (source, page_number)
            for page_start, page_end, source, page_number in page_ranges
            if page_start < end and page_end > start
        ]
        pages = [page_number for _, page_number in matching_pages]
        sources = sorted({source for source, _ in matching_pages})
        documents.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "chunk": chunk_number,
                    "source": sources,
                    "file_name": [Path(source).name for source in sources],
                    "pages": pages,
                    "page_start": pages[0] if pages else None,
                    "page_end": pages[-1] if pages else None,
                    "characters": len(chunk_text),
                },
            )
        )
    return documents


def create_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_and_save_vector_db(
    pdf_directory: Path = DEFAULT_PDF_DIRECTORY,
    vector_directory: Path = DEFAULT_VECTOR_DIRECTORY,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> FAISS:
    text, page_ranges = extract_pdf_text(pdf_directory)
    chunks = create_fixed_size_chunks(
        text, page_ranges, chunk_size, chunk_overlap
    )
    vector_store = FAISS.from_documents(chunks, create_embeddings(model_name))
    vector_store.save_local(str(vector_directory))
    return vector_store


def load_vector_db(
    vector_directory: Path = DEFAULT_VECTOR_DIRECTORY,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> FAISS:
    """Load a previously persisted FAISS index created by this script.

    FAISS stores the document metadata in a pickle file. Only load indexes
    from a directory you created or otherwise fully trust.
    """
    if not (vector_directory / "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found in {vector_directory}. "
            "Run this script to create it first."
        )
    return FAISS.load_local(
        str(vector_directory),
        create_embeddings(model_name),
        allow_dangerous_deserialization=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a persistent FAISS database with cross-page chunks."
    )
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIRECTORY)
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIRECTORY)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    vector_store = build_and_save_vector_db(
        pdf_directory=args.pdf_dir,
        vector_directory=args.vector_dir,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunk_sizes = {
        len(vector_store.docstore.search(document_id).page_content)
        for document_id in vector_store.index_to_docstore_id.values()
    }
    print(
        f"FAISS database created at '{args.vector_dir}' with "
        f"{len(vector_store.index_to_docstore_id)} chunks. "
        f"Chunk sizes: {sorted(chunk_sizes)}"
    )


if __name__ == "__main__":
    main()
