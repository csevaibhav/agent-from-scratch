"""
Step 5: Error handling & guardrails -- built on step 4's planning agent.

New concepts vs step 4:
  - TIMEOUTS: both the LLM call and each tool call run under a hard time
    limit. Without this, a hung network call or an infinite loop inside a
    tool freezes your entire agent with no way to recover.
  - FAILURE CLASSIFICATION: this is the fix for the exact bug you found by
    hand in step 4 -- retrying a "3335 > 75 unsupported expression" error
    3 times did nothing, because that failure was PERMANENT (a capability
    gap), not TRANSIENT (a temporary glitch). We now classify errors and
    only retry the ones retrying can actually fix.
  - GRACEFUL FAILURE REPORTING: when a subtask exhausts its retries, or the
    whole agent hits its iteration cap, we return a clear structured
    report of what succeeded and what didn't -- never a silent None,
    never an unhandled crash.

Setup:
    ollama pull llama3.1
    pip install ollama ddgs

Run:
    python step5_guardrails.py
"""

import ast
import operator
import concurrent.futures
from ollama import chat

MODEL = "llama3.1"
MAX_RETRIES_PER_SUBTASK = 2
MAX_ITERATIONS_PER_SUBTASK = 4
LLM_CALL_TIMEOUT_SECONDS = 60
TOOL_CALL_TIMEOUT_SECONDS = 15


# ---------------------------------------------------------------------------
# TOOLS (calculator, extended with comparisons from step 4's fix)
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
    raise ValueError("UNSUPPORTED_EXPRESSION")  # tagged so we can classify it below


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
            "description": "Evaluate a basic arithmetic expression, including comparisons (e.g. '3335 > 75').",
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
# TIMEOUT WRAPPER
#
# Both the model call and tool calls run inside a thread with a hard
# deadline. If either hangs past the timeout, we get a controlled
# TimeoutError instead of the whole script freezing indefinitely.
# ---------------------------------------------------------------------------
def run_with_timeout(fn, timeout_seconds, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"{fn.__name__} exceeded {timeout_seconds}s timeout")


# ---------------------------------------------------------------------------
# FAILURE CLASSIFICATION
#
# This is the direct fix for the step 4 bug: not all failures deserve a
# retry. We tag errors with a short reason code (see calculator() above)
# and classify each one as:
#   - "transient"  -> worth retrying as-is (timeouts, temporary glitches)
#   - "permanent"  -> retrying with the SAME input will never succeed
#                      (a capability gap, e.g. an unsupported operation)
# Real systems often go further and, on a "permanent" failure, ask the
# model to try a DIFFERENT approach rather than just giving up -- that's
# a natural next extension once this classification exists.
# ---------------------------------------------------------------------------
PERMANENT_ERROR_TAGS = ("UNSUPPORTED_EXPRESSION", "DIVISION_BY_ZERO")
TRANSIENT_ERROR_TAGS = ("TIMEOUT", "CONNECTION")


def classify_failure(error_message: str) -> str:
    if any(tag in error_message for tag in PERMANENT_ERROR_TAGS):
        return "permanent"
    if any(tag in error_message for tag in TRANSIENT_ERROR_TAGS):
        return "transient"
    # Unknown error shape -- default to transient so we at least try once
    # more, but this is a deliberate judgment call, not a certainty.
    return "transient"


# ---------------------------------------------------------------------------
# PLANNING (unchanged from step 4)
# ---------------------------------------------------------------------------
def make_plan(user_request: str) -> list[str]:
    import json
    planning_prompt = f"""Break the following request into an ordered list of \
concrete subtasks. Respond with ONLY a JSON array of strings, nothing else.

Example output format:
["Calculate 12 * 4", "Calculate 50 - 8", "Add the two results together"]

Request: {user_request}"""

    try:
        response = run_with_timeout(
            chat, LLM_CALL_TIMEOUT_SECONDS,
            model=MODEL, messages=[{"role": "user", "content": planning_prompt}],
        )
    except TimeoutError as e:
        print(f"[PLANNING] Timed out: {e}. Falling back to single-step plan.")
        return [user_request]

    raw = response["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        plan = json.loads(raw)
        if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
            return plan
    except json.JSONDecodeError:
        pass

    print(f"[PLANNING] Could not parse plan JSON, falling back to single-step.")
    return [user_request]


# ---------------------------------------------------------------------------
# EXECUTION -- now timeout-protected and error-classified.
# ---------------------------------------------------------------------------
def execute_subtask(subtask: str, prior_results: list[dict]) -> tuple[bool, str, str]:
    """Returns (success, result_text, failure_type).
    failure_type is "" on success, else "transient" or "permanent"."""

    if prior_results:
        context = "\n".join(f"- {r['subtask']}: {r['result']}" for r in prior_results)
        prompt = f"Context from previous subtasks:\n{context}\n\nNow do: {subtask}"
    else:
        prompt = subtask

    messages = [{"role": "user", "content": prompt}]

    for _ in range(MAX_ITERATIONS_PER_SUBTASK):
        try:
            response = run_with_timeout(
                chat, LLM_CALL_TIMEOUT_SECONDS, model=MODEL, messages=messages, tools=TOOLS
            )
        except TimeoutError as e:
            return False, str(e), "transient"

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
                output = run_with_timeout(fn, TOOL_CALL_TIMEOUT_SECONDS, **args)
            except TimeoutError as e:
                return False, str(e), "transient"
            except ValueError as e:
                # Tool functions raise ValueError with a tagged message
                # (see calculator() above) -- classify and bail immediately,
                # no point continuing this subtask's tool-use loop.
                return False, str(e), classify_failure(str(e))

            messages.append({"role": "tool", "content": output})

    return False, "Exceeded max iterations without a final answer.", "transient"


# ---------------------------------------------------------------------------
# ORCHESTRATION -- retries only transient failures, gives up immediately
# on permanent ones, and always produces a clear final report.
# ---------------------------------------------------------------------------
def run_agent(user_request: str):
    print(f"[PLANNING] Breaking down: {user_request!r}")
    plan = make_plan(user_request)
    print(f"[PLAN] {len(plan)} subtasks:")
    for i, task in enumerate(plan, 1):
        print(f"  {i}. {task}")

    results = []
    for i, subtask in enumerate(plan, 1):
        print(f"\n--- Subtask {i}/{len(plan)}: {subtask} ---")

        attempt = 0
        success, result, failure_type = False, None, ""
        while attempt <= MAX_RETRIES_PER_SUBTASK and not success:
            if attempt > 0:
                print(f"[RETRY] Attempt {attempt + 1} for subtask {i}")
            success, result, failure_type = execute_subtask(subtask, results)
            attempt += 1

            # This is the actual guardrail: stop retrying immediately if
            # the failure is classified as permanent, instead of burning
            # through all remaining retry attempts on something that will
            # never succeed unchanged.
            if not success and failure_type == "permanent":
                print(f"[GUARDRAIL] Permanent failure detected, skipping remaining retries.")
                break

        status = "OK" if success else "FAILED"
        print(f"[{status}] {result}")
        results.append({"subtask": subtask, "status": status, "result": result, "failure_type": failure_type})

    print("\n=== SUMMARY ===")
    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"{ok_count}/{len(results)} subtasks succeeded.\n")
    for r in results:
        tag = f" ({r['failure_type']} failure)" if r["status"] == "FAILED" else ""
        print(f"[{r['status']}]{tag} {r['subtask']} -> {r['result']}")

    return results


if __name__ == "__main__":
    # This request deliberately includes one impossible subtask (division
    # by zero) to demonstrate the permanent-failure guardrail: watch it
    # fail FAST on that one instead of burning 3 retries on it.
    run_agent(
        "Calculate 145 times 23, then calculate 50 divided by 0, "
        "then calculate 900 divided by 12."
    )