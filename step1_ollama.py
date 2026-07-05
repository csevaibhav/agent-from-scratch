"""
Step 1 (Ollama version): Minimal agent loop with ONE tool (calculator),
running 100% locally and free on your own GPU via Ollama.

Same exact loop logic as the Anthropic version -- Reason -> Act -> Observe --
just a different transport for talking to the model. This is the point:
once you understand the loop, swapping providers is a small, mechanical change.

Setup (one-time):
    1. Install Ollama: https://ollama.com/download  (Windows installer)
    2. Pull a tool-calling-capable model in a terminal:
           ollama pull llama3.1
       (qwen2.5 also works well and is a bit lighter on VRAM if llama3.1 is slow)
    3. pip install ollama

Run:
    python step1_ollama.py

No API key. No internet required after the model is downloaded. No bill.
"""

import json
from ollama import chat

MODEL = "llama3.1"  # swap to "qwen2.5" if this feels slow on your 4050


# ---------------------------------------------------------------------------
# 1. Tool schema -- identical shape/idea to the Anthropic version, just
#    using OpenAI-style function-calling format, which is what Ollama expects.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression, e.g. '12 * (4 + 3)'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python arithmetic expression",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# 2. The real function behind the tool -- unchanged from before.
# ---------------------------------------------------------------------------
def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


TOOL_FUNCTIONS = {"calculator": calculator}


# ---------------------------------------------------------------------------
# 3. The reasoning loop -- structurally the same as the Anthropic version.
#    The only real differences are:
#      - how we check "did the model ask for a tool?" (message.tool_calls)
#      - how we format the tool result back into the conversation
# ---------------------------------------------------------------------------
def run_agent(user_message: str, max_iterations: int = 5):
    messages = [{"role": "user", "content": user_message}]

    for step in range(max_iterations):
        print(f"\n--- iteration {step + 1} ---")

        response = chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        msg = response["message"]
        messages.append(msg)  # keep the model's own turn in history

        # If the model didn't ask for any tool calls, it's giving a final answer.
        if not msg.get("tool_calls"):
            print(f"\n[FINAL ANSWER] {msg['content']}")
            return msg["content"]

        # Otherwise, run every tool call it asked for and report results back.
        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            args = call["function"]["arguments"]  # already a dict with Ollama

            print(f"[TOOL CALL] {name}({args})")
            fn = TOOL_FUNCTIONS[name]
            output = fn(**args)
            print(f"[TOOL RESULT] {output}")

            # Ollama expects tool results as their own "tool" role message
            messages.append(
                {
                    "role": "tool",
                    "content": output,
                }
            )

    print("[STOPPED] Hit max_iterations without a final answer.")
    return None


if __name__ == "__main__":
    run_agent("What is (145 * 23) + 7, then divide that by 3?")