"""
FastAPI wrapper around the agent built in steps 1-7.

New concept: your agent stops being "a script you run manually" and
becomes "a service other programs can call over HTTP" -- this is the
actual shape production agents take. A frontend, a Slack bot, or another
service would call this API; they'd never run your .py file directly.

Setup:
    ollama pull llama3.1
    pip install ollama ddgs fastapi uvicorn

Run the server:
    uvicorn api:app --reload

Then either:
  - Open http://127.0.0.1:8000/docs for interactive API docs (FastAPI
    auto-generates this from your code -- try it right there in the browser)
  - Or send a request from a second terminal:
        curl -X POST http://127.0.0.1:8000/agent -H "Content-Type: application/json" -d "{\"request\": \"Calculate 145 * 23\"}"
"""

import ast
import operator
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import AsyncClient

MODEL = "llama3.1"
MAX_RETRIES_PER_SUBTASK = 2
MAX_ITERATIONS_PER_SUBTASK = 4
LLM_CALL_TIMEOUT_SECONDS = 60
LOG_FILE = Path("./agent_logs.jsonl")

client = AsyncClient()
app = FastAPI(title="Agent From Scratch API", version="1.0")


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
#
# Pydantic models define exactly what shape of JSON this API accepts and
# returns. FastAPI uses these to auto-validate incoming requests (a
# request missing "request" or sending the wrong type gets rejected
# automatically, before your code even runs) AND to auto-generate the
# /docs page. This replaces manually checking `if "request" not in body`.
# ---------------------------------------------------------------------------
class AgentRequest(BaseModel):
    request: str


class SubtaskResult(BaseModel):
    index: int
    task: str
    status: str
    result: str


class AgentResponse(BaseModel):
    session_id: str
    request: str
    succeeded: int
    total: int
    duration_seconds: float
    subtasks: list[SubtaskResult]


# ---------------------------------------------------------------------------
# LOGGING (same as step 7, unchanged)
# ---------------------------------------------------------------------------
def log_event(session_id: str, event_type: str, **fields):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "event": event_type,
        **fields,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# TOOLS (unchanged from step 7)
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
    }
]
TOOL_FUNCTIONS = {"calculator": calculator}


def call_tool(session_id: str, name: str, args: dict) -> tuple[str, str]:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        log_event(session_id, "tool_call", tool=name, args=args, success=False, error="no such tool")
        return f"UNSUPPORTED_EXPRESSION: no such tool '{name}'", "permanent"

    start = time.monotonic()
    try:
        output = fn(**args)
        log_event(session_id, "tool_call", tool=name, args=args, success=True,
                   result=output, duration_seconds=round(time.monotonic() - start, 3))
        return output, ""
    except ValueError as e:
        msg = str(e)
        failure_type = "permanent" if any(
            tag in msg for tag in ("UNSUPPORTED_EXPRESSION", "DIVISION_BY_ZERO")
        ) else "transient"
        log_event(session_id, "tool_call", tool=name, args=args, success=False,
                   error=msg, failure_type=failure_type, duration_seconds=round(time.monotonic() - start, 3))
        return msg, failure_type


# ---------------------------------------------------------------------------
# PLANNING (unchanged logic from step 7, session_id threaded through for logging)
# ---------------------------------------------------------------------------
async def make_plan(session_id: str, user_request: str) -> list[dict]:
    prompt = f"""Break the following request into subtasks. For each subtask, \
list which OTHER subtasks (by index, 0-based) it depends on. Independent \
subtasks get an empty depends_on list.

Respond with ONLY a JSON array, in this exact shape:
[{{"index": 0, "task": "...", "depends_on": []}}]

Request: {user_request}"""

    try:
        response = await asyncio.wait_for(
            client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}]),
            timeout=LLM_CALL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log_event(session_id, "planning_failed", reason="timeout")
        return [{"index": 0, "task": user_request, "depends_on": []}]

    raw = response["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        plan = json.loads(raw)
        if isinstance(plan, list):
            log_event(session_id, "plan_created", request=user_request, subtask_count=len(plan), plan=plan)
            return plan
    except json.JSONDecodeError:
        pass

    log_event(session_id, "planning_failed", reason="unparseable_json", raw_output=raw)
    return [{"index": 0, "task": user_request, "depends_on": []}]


# ---------------------------------------------------------------------------
# EXECUTION (unchanged logic from step 7)
# ---------------------------------------------------------------------------
async def execute_subtask(session_id: str, task_desc: str, prior_results: dict) -> tuple[bool, str, str]:
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
            output, failure_type = call_tool(session_id, call["function"]["name"], call["function"]["arguments"])
            if failure_type:
                return False, output, failure_type
            messages.append({"role": "tool", "content": output})

    return False, "Exceeded max iterations.", "transient"


async def run_subtask_with_retries(session_id: str, index: int, task_desc: str, prior_results: dict):
    start = time.monotonic()
    log_event(session_id, "subtask_started", index=index, task=task_desc)

    attempt = 0
    success, result, failure_type = False, None, ""
    while attempt <= MAX_RETRIES_PER_SUBTASK and not success:
        if attempt > 0:
            log_event(session_id, "subtask_retry", index=index, task=task_desc, attempt=attempt + 1)
        success, result, failure_type = await execute_subtask(session_id, task_desc, prior_results)
        attempt += 1
        if not success and failure_type == "permanent":
            break

    log_event(session_id, "subtask_finished", index=index, task=task_desc, success=success,
               result=result, duration_seconds=round(time.monotonic() - start, 3))
    return success, result, failure_type


# ---------------------------------------------------------------------------
# CORE ORCHESTRATION -- same dependency-wave logic as step 6/7, now
# returning a structured result instead of just printing to a terminal.
# ---------------------------------------------------------------------------
async def run_agent(user_request: str) -> AgentResponse:
    session_id = str(uuid.uuid4())[:8]
    session_start = time.monotonic()
    log_event(session_id, "session_started", request=user_request)

    plan = await make_plan(session_id, user_request)
    completed: dict[int, dict] = {}
    remaining = {p["index"]: p for p in plan}

    while remaining:
        ready = [p for p in remaining.values() if all(d in completed for d in p["depends_on"])]
        if not ready:
            log_event(session_id, "session_stuck", remaining_indices=list(remaining.keys()))
            break

        outcomes = await asyncio.gather(
            *[run_subtask_with_retries(session_id, p["index"], p["task"], completed) for p in ready]
        )
        for p, (success, result, failure_type) in zip(ready, outcomes):
            completed[p["index"]] = {
                "task": p["task"],
                "status": "OK" if success else "FAILED",
                "result": result,
            }
            del remaining[p["index"]]

    duration = round(time.monotonic() - session_start, 3)
    ok_count = sum(1 for r in completed.values() if r["status"] == "OK")
    log_event(session_id, "session_finished", total_subtasks=len(plan), succeeded=ok_count, duration_seconds=duration)

    return AgentResponse(
        session_id=session_id,
        request=user_request,
        succeeded=ok_count,
        total=len(plan),
        duration_seconds=duration,
        subtasks=[
            SubtaskResult(index=i, task=r["task"], status=r["status"], result=r["result"])
            for i, r in sorted(completed.items())
        ],
    )


# ---------------------------------------------------------------------------
# THE ACTUAL HTTP ENDPOINTS -- this is the new part this file adds.
# Everything above is just your existing agent, unchanged in behavior.
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Simple liveness check -- confirms the API process is up and responding."""
    return {"status": "ok"}


@app.post("/agent", response_model=AgentResponse)
async def run_agent_endpoint(payload: AgentRequest):
    """Submit a request; the agent plans it, executes subtasks (concurrently
    where possible), and returns a structured result -- no need to run a
    .py file by hand or read terminal output."""
    if not payload.request.strip():
        raise HTTPException(status_code=400, detail="request must not be empty")
    return await run_agent(payload.request)


@app.get("/logs/recent")
async def recent_logs(limit: int = 20):
    """Returns the most recent log entries -- a lightweight way to inspect
    agent behavior over HTTP instead of opening agent_logs.jsonl by hand."""
    if not LOG_FILE.exists():
        return {"entries": []}
    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    recent = [json.loads(line) for line in lines[-limit:]]
    return {"entries": recent}