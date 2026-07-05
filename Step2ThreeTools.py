"""
Step 2: Three tools + manual dispatch, still running locally on Ollama.

New concepts vs step 1:
  - Dispatch TABLE (dict) instead of if/elif -- this is what every agent
    framework does under the hood when you use @tool decorators.
  - Safer calculator (no more raw eval on model-controlled input).
  - The model must now CHOOSE the right tool for the job, not just use
    the one tool it has. Watch the [TOOL CALL] lines to see it decide.

Setup (if you haven't already):
    ollama pull llama3.1
    pip install ollama duckduckgo-search

Run:
    python step2_three_tools.py
"""

import ast
import operator
from pathlib import Path
from ollama import chat
from ddgs import DDGS

MODEL = "llama3.1"


# ---------------------------------------------------------------------------
# TOOL 1: Calculator -- now using a safe AST-based evaluator instead of
# raw eval(). This only allows numbers and basic math operators; anything
# else (e.g. an attempt to import os or call a function) raises an error
# instead of silently executing. Never trust eval() on model-generated
# strings once you touch real files, APIs, or user data.
# ---------------------------------------------------------------------------
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ---------------------------------------------------------------------------
# TOOL 2: Web search -- real search results via DuckDuckGo, no API key
# needed. We trim to top 3 results and keep snippets short so we don't
# flood the model's small context window with noise.
# ---------------------------------------------------------------------------
def web_search(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(f"- {r['title']}: {r['body'][:200]}")
        return "\n".join(formatted)
    except Exception as e:
        return f"Error performing search: {e}"


# ---------------------------------------------------------------------------
# TOOL 3: File reader -- reads a local text file. Restricted to a single
# "sandbox" folder so the model can never be tricked into reading files
# outside of where you intend (e.g. no walking up to C:\Users\...\secrets).
# This kind of boundary-setting is a real production concern, not
# over-engineering for a toy project.
# ---------------------------------------------------------------------------
SANDBOX_DIR = Path("./sandbox_files").resolve()
SANDBOX_DIR.mkdir(exist_ok=True)


def file_reader(filename: str) -> str:
    try:
        target = (SANDBOX_DIR / filename).resolve()
        # Make sure the resolved path is still inside SANDBOX_DIR --
        # blocks tricks like "../../../Windows/System32/something".
        if SANDBOX_DIR not in target.parents and target != SANDBOX_DIR:
            return "Error: access outside the sandbox folder is not allowed."
        if not target.exists():
            return f"Error: file '{filename}' not found in sandbox_files/."
        return target.read_text(encoding="utf-8")[:2000]  # cap length
    except Exception as e:
        return f"Error reading file: {e}"


# ---------------------------------------------------------------------------
# Tool schemas -- one entry per tool, in the OpenAI-style format Ollama expects.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression, e.g. '(12 + 3) * 4'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A math expression"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, or facts you don't know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_reader",
            "description": "Read the contents of a text file from the local sandbox_files/ folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file to read"}
                },
                "required": ["filename"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# THE DISPATCH TABLE -- this is the core new idea in this step.
# Instead of a chain of if/elif checking block.function.name, we just look
# the function up by name in this dict and call it. Adding a 4th tool later
# means adding one entry here and one entry in TOOLS above -- nothing else
# in the loop below has to change.
# ---------------------------------------------------------------------------
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "web_search": web_search,
    "file_reader": file_reader,
}


def run_agent(user_message: str, max_iterations: int = 6):
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
            if fn is None:
                # The model hallucinated a tool name that doesn't exist.
                # Don't crash -- report it back so the model can self-correct.
                output = f"Error: no such tool '{name}'"
            else:
                output = fn(**args)

            print(f"[TOOL RESULT] {output}")
            messages.append({"role": "tool", "content": output})

    print("[STOPPED] Hit max_iterations without a final answer.")
    return None


if __name__ == "__main__":
    # Create a sample file so the file_reader tool has something to find.
    sample = SANDBOX_DIR / "notes.txt"
    if not sample.exists():
        sample.write_text("Vaibhav's 12-week roadmap: Python -> async -> LLM APIs -> agents.")

    # This single prompt nudges the model to use all three tools, so you
    # can watch it pick the right one for each sub-task.
    run_agent(
        "Read the file notes.txt, then search the web for what MCP "
        "(Model Context Protocol) is in one sentence, then calculate 8 * 7."
    )