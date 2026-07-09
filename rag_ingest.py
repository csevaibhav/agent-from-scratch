"""
RAG Step 2a: Ingestion -- reads every .txt file from knowledge_base/,
chunks it, embeds it, and stores it in a PERSISTENT ChromaDB database on
disk (chroma_db/ folder). Unlike step 1, this survives between runs --
you ingest once, then query many times without re-embedding anything.

Setup:
    ollama pull nomic-embed-text
    pip install chromadb

Run:
    python rag_ingest.py

This creates knowledge_base/ (put your .txt files there) and chroma_db/
(the persistent vector store -- don't edit this by hand, it's managed
by ChromaDB). Re-run this script any time you add or change documents
in knowledge_base/ -- it re-ingests everything.
"""

from pathlib import Path
import ollama
import chromadb

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

KNOWLEDGE_BASE_DIR = Path("./knowledge_base")
CHROMA_DB_DIR = Path("./chroma_db")  # persistent storage lives here

KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)

# PersistentClient writes to disk instead of RAM -- this is the ONE
# change from step 1's chromadb.Client() that makes everything survive
# between separate script runs.
client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
collection = client.get_or_create_collection(name="documents")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


def ingest_file(filepath: Path):
    text = filepath.read_text(encoding="utf-8")
    doc_id = filepath.stem
    chunks = chunk_text(text)

    # Remove any existing chunks for this doc first, so re-running after
    # editing a file doesn't leave stale duplicate chunks behind.
    existing = collection.get(where={"source": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    for i, chunk in enumerate(chunks):
        vector = embed(chunk)
        collection.add(
            ids=[f"{doc_id}-chunk-{i}"],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{"source": doc_id, "chunk_index": i}],
        )
    print(f"[INGEST] {filepath.name} -> {len(chunks)} chunks")


if __name__ == "__main__":
    txt_files = list(KNOWLEDGE_BASE_DIR.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {KNOWLEDGE_BASE_DIR}/ -- creating a sample file to get started.")
        sample = KNOWLEDGE_BASE_DIR / "sample.txt"
        sample.write_text(
            "This project is an AI agent built from scratch in Python, without "
            "using frameworks like LangChain or CrewAI. It covers a ReAct "
            "reasoning loop, manual tool dispatch, short and long term memory, "
            "multi-step planning with dependency-aware execution, async "
            "concurrent execution, error handling with failure classification, "
            "and structured observability logging. It was later extended with "
            "a FastAPI wrapper and an MCP server connected to Claude Desktop.",
            encoding="utf-8",
        )
        txt_files = [sample]

    for f in txt_files:
        ingest_file(f)

    print(f"\n[DONE] {collection.count()} total chunks stored in {CHROMA_DB_DIR}/")