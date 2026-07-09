"""
RAG Step 2b: The agent, now with search_documents as a real tool -- same
dispatch pattern from steps 2-5 (a dict mapping tool names to functions),
just with a new entry that queries your persistent knowledge base instead
of the live web or a calculator.

This is the actual point of building RAG by hand instead of using a
framework: retrieval isn't special-cased anywhere in the agent loop.
It's just another tool the model can choose to call, exactly like
calculator or web_search. The MODEL decides when your own documents
are relevant to the question -- you don't have to hardcode "always
search documents first."

Setup:
    ollama pull llama3.1
    ollama pull nomic-embed-text
    pip install ollama chromadb
    python rag_ingest.py        # run this first to populate chroma_db/

Run:
    python rag_agent.py
"""

import ast
import operator
from pathlib import Path
from ollama import chat
import chromadb
import ollama as ollama_module

MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"
CHROMA_DB_DIR = Path("./chroma_db")

# Same PersistentClient as the ingestion script -- pointing at the SAME
# folder means this agent reads exactly what rag_ingest.py wrote there.
client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
collection = client.get_or_create_collection(name="documents")


# ---------------------------------------------------------------------------
# THE RAG TOOL -- embeds the query, asks ChromaDB for the closest chunks,
# and returns them as plain text for the model to read and reason over.
# This is retrieval ONLY -- the tool doesn't answer the question itself,
# it just finds relevant material. The MODEL does the actual answering,
# using these chunks as context. That division of labor is the "A" and
# "G" in RAG: Retrieval finds it, Generation answers with it.
# ---------------------------------------------------------------------------
def search_documents(query: str, top_k: int = 3) -> str:
    if collection.count() == 0:
        return "No documents have been ingested yet. Run rag_ingest.py first."

    query_vector = ollama_module.embed(model=EMBED_MODEL, input=query)["embeddings"][0]
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    if not results["documents"][0]:
        return "No relevant documents found."

    formatted = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        formatted.append(f"[Source: {meta['source']}] {doc}")
    return "\n\n".join(formatted)


# ---------------------------------------------------------------------------
# Calculator, unchanged from earlier steps -- kept here so this agent has
# more than one tool, letting you watch the model CHOOSE between them.
# ---------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _ev(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_ev(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    try:
        return str(_ev(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the local knowledge base for information relevant to a "
                "question. Use this whenever the question might be answered by "
                "documents the user has provided, rather than general knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to search for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "calculator": calculator,
}


def run_agent(user_message: str, max_iterations: int = 5):
    messages = [{"role": "user", "content": user_message}]

    for step in range(max_iterations):
        print(f"\n--- iteration {step + 1} ---")

        response = chat(model=MODEL, messages=messages, tools=TOOLS)
        msg = response["message"]
        messages.append(msg)

        if not msg.get("tool_calls"):
            print(f"\n[FINAL ANSWER] {msg['content']}")
            return msg["content"]

        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            print(f"[TOOL CALL] {name}({args})")

            fn = TOOL_FUNCTIONS.get(name)
            output = fn(**args) if fn else f"Error: no such tool '{name}'"

            print(f"[TOOL RESULT] {output[:200]}{'...' if len(output) > 200 else ''}")
            messages.append({"role": "tool", "content": output})

    print("[STOPPED] Hit max_iterations without a final answer.")
    return None


if __name__ == "__main__":
    run_agent("What is 47 * 12?")