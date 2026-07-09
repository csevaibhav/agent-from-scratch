"""
Step 8: RAG integration -- search_documents added as a real tool alongside
calculator, wired into the FULL planning + async + guardrails + observability
agent from step 7. This is the actual integration point promised back in
rag_agent.py: the same retrieval mechanism, now available to an agent that
can plan multi-step requests and run independent subtasks concurrently.

Nothing about the planning/execution/logging logic changes to support RAG --
that's the whole point of having built tools as a generic, pluggable
mechanism since step 2. search_documents is just one more entry in
TOOL_FUNCTIONS; the orchestration code doesn't know or care that it happens
to do a vector similarity search instead of arithmetic.

Setup:
    ollama pull llama3.1
    ollama pull nomic-embed-text
    pip install ollama chromadb
    python rag_ingest.py        # populate chroma_db/ before running this

Run:
    python step8_rag_agent.py

Then inspect the log same as step 7:
    python analyze_logs.py
"""

import ast
import operator
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from ollama import AsyncClient
import ollama as ollama_module
import chromadb

MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"
MAX_RETRIES_PER_SUBTASK = 2
MAX_ITERATIONS_PER_SUBTASK = 4
LLM_CALL_TIMEOUT_SECONDS = 60
LOG_FILE = Path("./agent_logs.jsonl")
CHROMA_DB_DIR = Path("./chroma_db")

client = AsyncClient()
SESSION_ID = str(uuid.uuid4())[:8]

# Same persistent store rag_ingest.py writes to -- this agent only READS
# from it; ingestion stays a separate, deliberate step (you don't want
# every agent run silently re-embedding your whole knowledge base).
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
collection = chroma_client.get_or_create_collection(name="documents")


# ---------------------------------------------------------------------------
# THE LOGGER
#
# One function, append-only, one JSON object per line. Deliberately dead
# simple -- this is the same idea a proper logging library (Python's
# `logging` module, or a cloud log service) does with far more features,
# but the core mechanic (structured event + timestamp + context, written
# somewhere durable) is identical.
# ---------------------------------------------------------------------------
def log_event(event_type: str, **fields):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": SESSION_ID,
        "event": event_type,
        **fields,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# TOOLS -- unchanged from step 6, except calculator calls are now timed
# and logged individually.
# ---------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
_CMP_OPS = {ast.Gt: operator.gt, ast.Lt: operator.lt, ast.GtE: operator.ge,
            ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne}


def _ev(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_ev(node.operand))
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return _CMP_OPS[type(node.ops[0])](_ev(node.left), _ev(node.comparators[0]))
    raise ValueError("UNSUPPORTED_EXPRESSION")


def calculator(expression: str) -> str:
    try:
        return str(_ev(ast.parse(expression, mode="eval").body))
    except ZeroDivisionError:
        raise ValueError("DIVISION_BY_ZERO")
    except Exception as e:
        raise ValueError(f"UNSUPPORTED_EXPRESSION: {e}")


# ---------------------------------------------------------------------------
# search_documents -- the new tool this step adds. Same shape as calculator:
# takes plain arguments, returns a string, raises ValueError on failure so
# it plugs into the exact same call_tool()/classification machinery below
# with zero special-casing.
# ---------------------------------------------------------------------------
def search_documents(query: str, top_k: int = 3) -> str:
    if collection.count() == 0:
        raise ValueError("NO_DOCUMENTS_INGESTED: run rag_ingest.py first")

    query_vector = ollama_module.embed(model=EMBED_MODEL, input=query)["embeddings"][0]
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    if not results["documents"][0]:
        return "No relevant documents found."

    formatted = [
        f"[Source: {meta['source']}] {doc}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    return "\n\n".join(formatted)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression, including comparisons.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the local knowledge base for information relevant to a "
                "question. Use this when the question might be answered by "
                "documents already provided, rather than general knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to search for"}},
                "required": ["query"],
            },
        },
    },
]
TOOL_FUNCTIONS = {"calculator": calculator, "search_documents": search_documents}


def call_tool(name: str, args: dict) -> tuple[str, str]:
    """Runs a tool, logs the call with duration, returns (output, failure_type)."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        log_event("tool_call", tool=name, args=args, success=False, error="no such tool")
        return f"UNSUPPORTED_EXPRESSION: no such tool '{name}'", "permanent"

    start = time.monotonic()
    try:
        output = fn(**args)
        duration = round(time.monotonic() - start, 3)
        log_event("tool_call", tool=name, args=args, success=True,
                   result=output, duration_seconds=duration)
        return output, ""
    except ValueError as e:
        duration = round(time.monotonic() - start, 3)
        msg = str(e)
        failure_type = "permanent" if any(
            tag in msg for tag in ("UNSUPPORTED_EXPRESSION", "DIVISION_BY_ZERO", "NO_DOCUMENTS_INGESTED")
        ) else "transient"
        log_event("tool_call", tool=name, args=args, success=False,
                   error=msg, failure_type=failure_type, duration_seconds=duration)
        return msg, failure_type


# ---------------------------------------------------------------------------
# PLANNING -- now logs the plan itself and how long planning took.
# ---------------------------------------------------------------------------
async def make_plan(user_request: str) -> list[dict]:
    prompt = f"""Break the following request into subtasks. For each subtask, \
list which OTHER subtasks (by index, 0-based) it depends on. Independent \
subtasks get an empty depends_on list.

Respond with ONLY a JSON array, in this exact shape:
[{{"index": 0, "task": "...", "depends_on": []}}]

Request: {user_request}"""

    start = time.monotonic()
    try:
        response = await asyncio.wait_for(
            client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}]),
            timeout=LLM_CALL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log_event("planning_failed", reason="timeout", duration_seconds=round(time.monotonic() - start, 3))
        return [{"index": 0, "task": user_request, "depends_on": []}]

    duration = round(time.monotonic() - start, 3)
    raw = response["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        plan = json.loads(raw)
        if isinstance(plan, list):
            log_event("plan_created", request=user_request, subtask_count=len(plan),
                       plan=plan, duration_seconds=duration)
            return plan
    except json.JSONDecodeError:
        pass

    log_event("planning_failed", reason="unparseable_json", raw_output=raw, duration_seconds=duration)
    return [{"index": 0, "task": user_request, "depends_on": []}]


# ---------------------------------------------------------------------------
# EXECUTION -- logs subtask start/end and every retry.
# ---------------------------------------------------------------------------
async def execute_subtask(task_desc: str, prior_results: dict) -> tuple[bool, str, str]:
    if prior_results:
        context = "\n".join(f"- {v['task']}: {v['result']}" for v in prior_results.values())
        prompt = f"Context from dependencies already completed:\n{context}\n\nNow do: {task_desc}"
    else:
        prompt = task_desc

    messages = [{"role": "user", "content": prompt}]

    for _ in range(MAX_ITERATIONS_PER_SUBTASK):
        try:
            response = await asyncio.wait_for(
                client.chat(model=MODEL, messages=messages, tools=TOOLS),
                timeout=LLM_CALL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return False, "LLM call TIMEOUT", "transient"

        msg = response["message"]
        messages.append(msg)

        if not msg.get("tool_calls"):
            return True, msg["content"], ""

        for call in msg["tool_calls"]:
            output, failure_type = call_tool(call["function"]["name"], call["function"]["arguments"])
            if failure_type:
                return False, output, failure_type
            messages.append({"role": "tool", "content": output})

    return False, "Exceeded max iterations.", "transient"


async def run_subtask_with_retries(index: int, task_desc: str, prior_results: dict) -> tuple[bool, str, str]:
    start = time.monotonic()
    log_event("subtask_started", index=index, task=task_desc)

    attempt = 0
    success, result, failure_type = False, None, ""
    while attempt <= MAX_RETRIES_PER_SUBTASK and not success:
        if attempt > 0:
            log_event("subtask_retry", index=index, task=task_desc, attempt=attempt + 1)
        success, result, failure_type = await execute_subtask(task_desc, prior_results)
        attempt += 1
        if not success and failure_type == "permanent":
            break

    duration = round(time.monotonic() - start, 3)
    log_event("subtask_finished", index=index, task=task_desc, success=success,
               result=result, failure_type=failure_type, attempts=attempt,
               duration_seconds=duration)
    return success, result, failure_type


# ---------------------------------------------------------------------------
# ORCHESTRATION -- same dependency-wave logic as step 6, now with a
# session-level start/end log so you can see total wall-clock time and
# overall success rate per run.
# ---------------------------------------------------------------------------
async def run_agent(user_request: str):
    session_start = time.monotonic()
    log_event("session_started", request=user_request)
    print(f"[SESSION {SESSION_ID}] {user_request!r}")

    plan = await make_plan(user_request)
    print(f"[PLAN] {len(plan)} subtasks")

    completed: dict[int, dict] = {}
    remaining = {p["index"]: p for p in plan}
    wave_num = 0

    while remaining:
        wave_num += 1
        ready = [p for p in remaining.values() if all(d in completed for d in p["depends_on"])]
        if not ready:
            log_event("session_stuck", remaining_indices=list(remaining.keys()))
            print(f"[STUCK] Unmet dependencies: {list(remaining.keys())}")
            break

        print(f"\n=== Wave {wave_num}: {[p['index'] for p in ready]} ===")
        outcomes = await asyncio.gather(
            *[run_subtask_with_retries(p["index"], p["task"], completed) for p in ready]
        )
        for p, (success, result, failure_type) in zip(ready, outcomes):
            status = "OK" if success else "FAILED"
            print(f"[{status}] {p['task']} -> {result}")
            completed[p["index"]] = {"task": p["task"], "status": status, "result": result}
            del remaining[p["index"]]

    total_duration = round(time.monotonic() - session_start, 3)
    ok_count = sum(1 for r in completed.values() if r["status"] == "OK")
    log_event("session_finished", total_subtasks=len(plan), succeeded=ok_count,
               duration_seconds=total_duration)

    print(f"\n=== SUMMARY === {ok_count}/{len(plan)} succeeded in {total_duration}s")
    print(f"[LOG] Full structured trace written to {LOG_FILE} (session {SESSION_ID})")
    return completed


if __name__ == "__main__":
    asyncio.run(run_agent(
        "Search the documents for what error handling approach this project uses, "
        "separately calculate 900 / 12, then summarize both results together."
    ))