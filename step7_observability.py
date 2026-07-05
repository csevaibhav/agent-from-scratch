"""
Step 7: Observability -- structured, queryable logging on top of step 6's
async planning agent.

New concepts vs step 6:
  - Every meaningful event (plan created, subtask started/finished, tool
    called, retry, failure) is written as ONE JSON object per line to
    agent_logs.jsonl -- this format is called "JSON Lines" (JSONL) and is
    the standard for logs: easy to append to, easy to parse line-by-line
    even if the file is huge, and every major log platform (Datadog,
    CloudWatch, etc.) expects roughly this shape.
  - Every event carries a session_id (one per run of the script) and a
    timestamp, so you can later filter "just this run" or "just today."
  - Durations are measured and logged, not just guessed from watching
    the terminal -- this is what makes "what's slow?" an answerable
    question instead of a vibe.
  - print() statements are KEPT alongside logging, not replaced -- you
    still want human-readable output while developing. Logging is an
    ADDITIONAL layer for later analysis, not a replacement for stdout.

Setup:
    ollama pull llama3.1
    pip install ollama ddgs

Run:
    python step7_observability.py

Then inspect the log:
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

MODEL = "llama3.1"
MAX_RETRIES_PER_SUBTASK = 2
MAX_ITERATIONS_PER_SUBTASK = 4
LLM_CALL_TIMEOUT_SECONDS = 60
LOG_FILE = Path("./agent_logs.jsonl")

client = AsyncClient()
SESSION_ID = str(uuid.uuid4())[:8]  # short id to tell separate runs apart in the log


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
            tag in msg for tag in ("UNSUPPORTED_EXPRESSION", "DIVISION_BY_ZERO")
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
        "Calculate 145 * 23, separately calculate 900 / 12, then calculate 50 / 0, "
        "then tell me the sum of the successful results."
    ))