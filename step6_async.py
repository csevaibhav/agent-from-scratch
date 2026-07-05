"""
Step 6: Async execution -- independent subtasks run CONCURRENTLY instead of
one-at-a-time, while subtasks that depend on earlier results still wait
for them, in the correct order.

New concepts vs step 5:
  - The PLANNING step now also outputs which subtasks depend on which
    others (as a list of indices), not just an ordered list of strings.
    This dependency info is what makes safe concurrency possible -- we
    can only run things in parallel once we know they don't need each
    other's output.
  - asyncio.gather() runs multiple subtasks at the same time and waits
    for all of them to finish, instead of awaiting them one by one.
  - The ollama client's async interface (AsyncClient) is used so the
    actual network/model calls don't block each other.

Why this matters concretely: in step 5's example, "145 * 23" and
"900 / 12" have zero relationship to each other. Sequentially, if each
takes ~3 seconds, that's 6 seconds. Concurrently, it's ~3 seconds total --
you're not waiting on the model twice for no reason.

Setup:
    ollama pull llama3.1
    pip install ollama ddgs

Run:
    python step6_async.py
"""

import ast
import operator
import asyncio
import json
import time
from ollama import AsyncClient

MODEL = "llama3.1"
MAX_RETRIES_PER_SUBTASK = 2
MAX_ITERATIONS_PER_SUBTASK = 4
LLM_CALL_TIMEOUT_SECONDS = 60
TOOL_CALL_TIMEOUT_SECONDS = 15

client = AsyncClient()


# ---------------------------------------------------------------------------
# TOOLS -- same calculator as step 5. Note tool functions themselves stay
# SYNCHRONOUS -- they're fast, CPU-only, and don't need to be async. Only
# I/O-bound things (LLM calls, network calls) benefit from async; wrapping
# a calculator in async would add complexity for zero benefit. Async is a
# tool for a specific problem (waiting on I/O), not a blanket upgrade.
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


# ---------------------------------------------------------------------------
# PLANNING -- now asks for dependencies too. Each subtask gets an index,
# and a "depends_on" list of earlier indices it needs results from. A
# subtask with an empty depends_on list can run concurrently with any
# other subtask that also has no unmet dependencies.
# ---------------------------------------------------------------------------
async def make_plan(user_request: str) -> list[dict]:
    prompt = f"""Break the following request into subtasks. For each subtask, \
also list which OTHER subtasks (by index, 0-based) it depends on -- i.e. \
needs the result of before it can run. Independent subtasks should have an \
empty depends_on list.

Respond with ONLY a JSON array, nothing else, in this exact shape:
[
  {{"index": 0, "task": "Calculate 12 * 4", "depends_on": []}},
  {{"index": 1, "task": "Calculate 50 - 8", "depends_on": []}},
  {{"index": 2, "task": "Add the two results together", "depends_on": [0, 1]}}
]

Request: {user_request}"""

    try:
        response = await asyncio.wait_for(
            client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}]),
            timeout=LLM_CALL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print("[PLANNING] Timed out, falling back to single-step plan.")
        return [{"index": 0, "task": user_request, "depends_on": []}]

    raw = response["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        plan = json.loads(raw)
        if isinstance(plan, list):
            return plan
    except json.JSONDecodeError:
        pass

    print("[PLANNING] Could not parse plan JSON, falling back to single-step.")
    return [{"index": 0, "task": user_request, "depends_on": []}]


# ---------------------------------------------------------------------------
# EXECUTION -- same logic as step 5, rewritten as an async function so it
# can run concurrently with OTHER subtasks via asyncio.gather() below.
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
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            fn = TOOL_FUNCTIONS.get(name)

            if fn is None:
                return False, f"UNSUPPORTED_EXPRESSION: no such tool '{name}'", "permanent"

            try:
                # Tool itself is sync and fast -- just call it directly,
                # no need to thread/await a plain calculator function.
                output = fn(**args)
            except ValueError as e:
                msg_str = str(e)
                failure_type = "permanent" if any(
                    tag in msg_str for tag in ("UNSUPPORTED_EXPRESSION", "DIVISION_BY_ZERO")
                ) else "transient"
                return False, msg_str, failure_type

            messages.append({"role": "tool", "content": output})

    return False, "Exceeded max iterations.", "transient"


async def run_subtask_with_retries(task_desc: str, prior_results: dict) -> tuple[bool, str, str]:
    attempt = 0
    success, result, failure_type = False, None, ""
    while attempt <= MAX_RETRIES_PER_SUBTASK and not success:
        if attempt > 0:
            print(f"[RETRY] Attempt {attempt + 1}: {task_desc}")
        success, result, failure_type = await execute_subtask(task_desc, prior_results)
        attempt += 1
        if not success and failure_type == "permanent":
            break
    return success, result, failure_type


# ---------------------------------------------------------------------------
# ORCHESTRATION -- executes the plan in DEPENDENCY WAVES:
#   Wave 1: every subtask with no unmet dependencies, run CONCURRENTLY
#   Wave 2: every subtask whose dependencies are now satisfied, concurrently
#   ...repeat until all subtasks are done (or stuck).
# This is the actual async payoff: within a wave, asyncio.gather() fires
# all the LLM calls at once instead of waiting for each one in turn.
# ---------------------------------------------------------------------------
async def run_agent(user_request: str):
    print(f"[PLANNING] Breaking down: {user_request!r}")
    plan = await make_plan(user_request)
    print(f"[PLAN] {len(plan)} subtasks:")
    for p in plan:
        dep_str = f" (depends on {p['depends_on']})" if p["depends_on"] else " (independent)"
        print(f"  {p['index']}. {p['task']}{dep_str}")

    completed: dict[int, dict] = {}
    remaining = {p["index"]: p for p in plan}

    wave_num = 0
    while remaining:
        wave_num += 1
        # A subtask is ready this wave if every index in its depends_on
        # list is already in `completed`.
        ready = [
            p for p in remaining.values()
            if all(dep in completed for dep in p["depends_on"])
        ]
        if not ready:
            print(f"[STUCK] Remaining subtasks have unmet dependencies: {list(remaining.keys())}")
            break

        ready_indices = [p["index"] for p in ready]
        print(f"\n=== Wave {wave_num}: running {len(ready)} subtask(s) concurrently: {ready_indices} ===")

        start = time.monotonic()
        # THIS is the actual concurrency: asyncio.gather() starts all
        # these coroutines and lets them run at the same time, rather
        # than awaiting each one to completion before starting the next.
        outcomes = await asyncio.gather(
            *[run_subtask_with_retries(p["task"], completed) for p in ready]
        )
        elapsed = time.monotonic() - start
        print(f"=== Wave {wave_num} finished in {elapsed:.1f}s ===")

        for p, (success, result, failure_type) in zip(ready, outcomes):
            status = "OK" if success else "FAILED"
            print(f"[{status}] {p['task']} -> {result}")
            completed[p["index"]] = {"task": p["task"], "status": status, "result": result}
            del remaining[p["index"]]

    print("\n=== SUMMARY ===")
    ok_count = sum(1 for r in completed.values() if r["status"] == "OK")
    print(f"{ok_count}/{len(plan)} subtasks succeeded.\n")
    for i in sorted(completed):
        r = completed[i]
        print(f"[{r['status']}] {r['task']} -> {r['result']}")

    return completed


if __name__ == "__main__":
    # 3 independent calculations + 1 that depends on all of them --
    # watch Wave 1 run 3 subtasks concurrently, then Wave 2 run the
    # dependent one alone once it has everything it needs.
    asyncio.run(run_agent(
        "Calculate 145 * 23, separately calculate 900 / 12, separately "
        "calculate 50 + 50, then tell me the sum of all three results."
    ))