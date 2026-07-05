"""
Step 4: Planning -- break a multi-step request into an explicit subtask list,
then execute each subtask (with its own tool-use loop), retrying on failure.

New concepts vs step 3:
  - A separate PLANNING call: before doing any work, ask the model to
    output a structured list of subtasks as JSON. This is a different
    "mode" than the normal reasoning loop -- we're asking it to think
    about the whole request, not react to one piece at a time.
  - EXECUTION per subtask: each subtask gets its own mini agent loop
    (same reasoning loop from steps 1-3), so a subtask can still use
    tools, memory, etc. -- planning sits ABOVE the existing loop, it
    doesn't replace it.
  - RETRY logic: if a subtask's execution looks like it failed, retry it
    a bounded number of times before giving up and moving on, instead of
    silently continuing with a broken result.

Setup:
    ollama pull llama3.1
    pip install ollama ddgs

Run:
    python step4_planning.py
"""

import json
from ollama import chat

MODEL = "llama3.1"
MAX_RETRIES_PER_SUBTASK = 2
MAX_ITERATIONS_PER_SUBTASK = 4


# ---------------------------------------------------------------------------
# Reuse the same tools from step 2/3 -- planning doesn't change what the
# agent can DO, only how it decides the order of operations.
# ---------------------------------------------------------------------------
def calculator(expression: str) -> str:
    import ast, operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
    cmp_ops = {ast.Gt: operator.gt, ast.Lt: operator.lt, ast.GtE: operator.ge,
               ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne}

    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            # supports simple two-sided comparisons like "3335 > 75"
            return cmp_ops[type(node.ops[0])](ev(node.left), ev(node.comparators[0]))
        raise ValueError("unsupported expression")

    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


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
# PLANNING PHASE
#
# We ask the model to respond with ONLY a JSON array of subtask strings --
# no prose, no explanation. This is a common pattern: when you need
# structured output you can parse programmatically, explicitly instruct
# the format AND give an example, then defensively parse the result
# (models -- especially small local ones -- sometimes wrap JSON in
# markdown fences or add a sentence before/after despite instructions).
# ---------------------------------------------------------------------------
def make_plan(user_request: str) -> list[str]:
    planning_prompt = f"""Break the following request into an ordered list of \
concrete subtasks. Respond with ONLY a JSON array of strings, nothing else \
-- no markdown fences, no explanation.

Example output format:
["Calculate 12 * 4", "Calculate 50 - 8", "Add the two results together"]

Request: {user_request}"""

    response = chat(model=MODEL, messages=[{"role": "user", "content": planning_prompt}])
    raw = response["message"]["content"].strip()

    # Defensive parsing: strip markdown fences if the model added them anyway.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    try:
        plan = json.loads(raw)
        if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
            return plan
    except json.JSONDecodeError:
        pass

    # Fallback: if parsing fails, treat the whole request as one subtask
    # rather than crashing -- a degraded plan is better than no plan.
    print(f"[PLANNING] Could not parse plan JSON, falling back to single-step. Raw output: {raw!r}")
    return [user_request]


# ---------------------------------------------------------------------------
# EXECUTION PHASE -- same reasoning loop as steps 1-3, just scoped to one
# subtask at a time instead of the whole user request.
# ---------------------------------------------------------------------------
def execute_subtask(subtask: str, prior_results: list[dict]) -> tuple[bool, str]:
    """Returns (success, result_text).

    prior_results carries forward what earlier subtasks produced, so a
    later subtask like "compare the two results" actually has the two
    results to compare instead of hallucinating them.
    """
    if prior_results:
        context = "\n".join(f"- {r['subtask']}: {r['result']}" for r in prior_results)
        prompt = (
            f"Context from previous subtasks already completed:\n{context}\n\n"
            f"Now do this subtask, using the context above where relevant: {subtask}"
        )
    else:
        prompt = subtask

    messages = [{"role": "user", "content": prompt}]

    for _ in range(MAX_ITERATIONS_PER_SUBTASK):
        response = chat(model=MODEL, messages=messages, tools=TOOLS)
        msg = response["message"]
        messages.append(msg)

        if not msg.get("tool_calls"):
            return True, msg["content"]

        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            fn = TOOL_FUNCTIONS.get(name)
            output = fn(**args) if fn else f"Error: no such tool '{name}'"

            # A simple failure signal: if the tool itself reported an error,
            # we treat this subtask attempt as failed so it can be retried.
            if output.startswith("Error"):
                return False, output

            messages.append({"role": "tool", "content": output})

    return False, "Exceeded max iterations without a final answer."


# ---------------------------------------------------------------------------
# ORCHESTRATION: plan -> execute each subtask -> retry failures -> report.
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
        success, result = False, None
        while attempt <= MAX_RETRIES_PER_SUBTASK and not success:
            if attempt > 0:
                print(f"[RETRY] Attempt {attempt + 1} for subtask {i}")
            success, result = execute_subtask(subtask, results)  # pass completed subtasks so far
            attempt += 1

        status = "OK" if success else "FAILED"
        print(f"[{status}] {result}")
        results.append({"subtask": subtask, "status": status, "result": result})

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"[{r['status']}] {r['subtask']} -> {r['result']}")

    return results


if __name__ == "__main__":
    run_agent(
        "Calculate 145 times 23, then separately calculate 900 divided by 12, "
        "then tell me which of those two results is bigger."
    )