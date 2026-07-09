"""
RAG Step 1: Chunking + Embedding + Retrieval -- the three core mechanics
of RAG, built by hand before wiring it into the agent.

Setup:
    ollama pull nomic-embed-text     # a small model dedicated to embeddings,
                                      # separate from llama3.1 which does reasoning
    pip install chromadb

Run:
    python rag_step1_retrieve.py

Concept map:
  CHUNKING  -> splitting long documents into smaller pieces. Why not embed
               a whole document at once? Because a 10-page doc embedded as
               ONE vector blurs together everything it discusses -- a
               question about paragraph 8 gets diluted by paragraphs 1-7
               and 9-10 all pulling the "meaning" in different directions.
               Smaller chunks = sharper, more specific matches.

  EMBEDDING -> converting text into a list of numbers (a vector) that
               captures its MEANING, not its exact words. This is what
               makes RAG different from Ctrl+F: "How do I stop the agent
               from retrying forever?" can match a chunk that says
               "guardrails prevent infinite retry loops" even though not
               a single word is shared -- because the *meaning* is close.

  RETRIEVAL -> given a question, embed IT the same way, then find which
               stored chunks have the closest vectors (closest = most
               similar meaning). ChromaDB does this nearest-neighbor
               search for you; you just give it vectors to compare.
"""

import ollama
import chromadb

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 500       # characters per chunk -- small enough to be specific,
CHUNK_OVERLAP = 50     # large enough to hold a complete thought
# CHUNK_OVERLAP: chunks share a bit of text at their boundary so a
# sentence that gets cut in half by a chunk boundary still appears
# WHOLE in at least one chunk.


# ---------------------------------------------------------------------------
# CHUNKING -- deliberately simple: fixed-size sliding window over raw text.
# Real systems often chunk on natural boundaries (paragraphs, headings)
# instead of a blind character count -- worth upgrading to later, but this
# makes the mechanism obvious first.
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# EMBEDDING -- one function call to Ollama, returns a vector (list of floats).
# nomic-embed-text is a SEPARATE, smaller model from llama3.1 -- embedding
# models are specialized for "represent meaning as numbers," not for
# generating text, so they're much faster and lighter than a full LLM.
# ---------------------------------------------------------------------------
def embed(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


# ---------------------------------------------------------------------------
# VECTOR STORE -- ChromaDB running in-memory (nothing persisted to disk in
# this step -- step 2 will make it persistent). It stores chunks alongside
# their embeddings, and does the nearest-neighbor search for you when you
# query it with a new embedding.
# ---------------------------------------------------------------------------
client = chromadb.Client()
collection = client.get_or_create_collection(name="documents")


def add_document(doc_id: str, text: str):
    chunks = chunk_text(text)
    print(f"[INGEST] '{doc_id}' split into {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        vector = embed(chunk)
        collection.add(
            ids=[f"{doc_id}-chunk-{i}"],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{"source": doc_id, "chunk_index": i}],
        )


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    query_vector = embed(query)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    matches = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        matches.append({"text": doc, "source": meta["source"], "distance": distance})
    return matches


if __name__ == "__main__":
    # A small fake "knowledge base" -- three unrelated documents, so we can
    # confirm retrieval actually picks the RIGHT one for a given question,
    # not just "the first thing in the database."
    add_document(
        "agent_architecture",
        "The agent uses a ReAct-style reasoning loop: the model reasons about "
        "what to do, takes an action by calling a tool, observes the result, "
        "and repeats until it has a final answer. Tool dispatch is handled "
        "by a dictionary mapping tool names to Python functions, which scales "
        "better than a long if/elif chain as more tools are added.",
    )
    add_document(
        "error_handling",
        "Failures are classified as either transient or permanent before "
        "deciding whether to retry. Transient failures, like a network "
        "timeout, are worth retrying since the same call might succeed "
        "on a second attempt. Permanent failures, like an unsupported "
        "operation or a division by zero, will fail identically no matter "
        "how many times you retry, so the guardrail logic skips remaining "
        "retry attempts immediately once a permanent failure is detected.",
    )
    add_document(
        "cricket_notes",
        "T20 cricket matches consist of a maximum of 20 overs per side. "
        "The format was introduced to produce faster-paced, more "
        "spectator-friendly matches compared to the longer One Day "
        "International and Test match formats.",
    )

    print("\n--- Query: 'Why does the agent stop retrying some errors immediately?' ---")
    for match in retrieve("Why does the agent stop retrying some errors immediately?"):
        print(f"[{match['source']}] (distance={match['distance']:.4f}) {match['text'][:100]}...")

    print("\n--- Query: 'How many overs are in a T20 match?' ---")
    for match in retrieve("How many overs are in a T20 match?"):
        print(f"[{match['source']}] (distance={match['distance']:.4f}) {match['text'][:100]}...")