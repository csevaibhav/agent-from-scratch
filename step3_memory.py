"""
Step 3: Memory -- short-term (conversation history) + long-term (persisted facts).

New concepts vs step 2:
  - Long-term memory as TWO NEW TOOLS ("remember" and "recall_memories")
    backed by a JSON file on disk. The model has to explicitly choose to
    save something -- it doesn't happen automatically, same as a human
    choosing to write something down vs just holding it in working memory.
  - Short-term memory TRIMMING -- conversation history grows every turn.
    Left unchecked, it eventually exceeds the model's context window and
    the whole agent breaks. We cap it here with a simple sliding window.

Setup:
    ollama pull llama3.1
    pip install ollama ddgs

Run:
    python step3_memory.py
"""

import json
from pathlib import Path
from ollama import chat

MODEL = "llama3.1"
MEMORY_FILE = Path("./long_term_memory.json")
MAX_HISTORY_MESSAGES = 20  # short-term memory cap -- see trim_history()


# ---------------------------------------------------------------------------
# LONG-TERM MEMORY: a tiny JSON-file-backed fact store.
#
# This is deliberately the simplest possible implementation -- a list of
# strings on disk. Real systems use a vector DB for semantic search over
# many facts, but the underlying idea is identical: persist something
# outside the conversation, then retrieve it later. Start here before you
# reach for a vector DB -- you'll understand WHY you need one once this
# simple version starts to strain (e.g. hundreds of facts, needing
# similarity search instead of "return everything").
# ---------------------------------------------------------------------------
def _load_memory() -> list[str]:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return []


def _save_memory(facts: list[str]) -> None:
    MEMORY_FILE.write_text(json.dumps(facts, indent=2), encoding="utf-8")


def remember(fact: str) -> str:
    facts = _load_memory()
    facts.append(fact)
    _save_memory(facts)
    return f"Saved to long-term memory: '{fact}'"


def recall_memories(query: str = "") -> str:
    facts = _load_memory()
    if not facts:
        return "No long-term memories stored yet."
    # Naive "search": just return everything. A vector DB is what you'd
    # swap this for once you have too many facts to dump in full each time.
    return "\n".join(f"- {f}" for f in facts)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save an important fact to permanent long-term memory, so it's "
                "available in future conversations, not just this one. Use this "
                "when the user shares something worth remembering long-term "
                "(preferences, ongoing projects, key facts about them)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember, written clearly and self-contained"}
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memories",
            "description": "Retrieve everything saved in long-term memory from past conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What you're trying to recall (currently unused, returns all memories)"}
                },
                "required": [],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "remember": remember,
    "recall_memories": recall_memories,
}


# ---------------------------------------------------------------------------
# SHORT-TERM MEMORY TRIMMING
#
# Every tool call, tool result, and assistant reply gets appended to
# `messages`. Left unbounded, a long-running agent session eventually
# exceeds the model's context window and either crashes or starts silently
# forgetting the START of the conversation instead of the middle.
#
# This is a simple sliding-window trim: always keep the very first message
# (the original user request -- often has critical context) plus the most
# recent N messages. Production systems often replace this with an actual
# LLM-generated summary of the trimmed-out middle -- we're keeping it
# simple here so the mechanism is obvious.
# ---------------------------------------------------------------------------
def trim_history(messages: list[dict]) -> list[dict]:
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    first = messages[0]
    recent = messages[-(MAX_HISTORY_MESSAGES - 1):]
    print(f"[MEMORY] Trimmed short-term history: {len(messages)} -> {len(recent) + 1} messages")
    return [first] + recent


def run_agent(user_message: str, max_iterations: int = 6):
    messages = [{"role": "user", "content": user_message}]

    for step in range(max_iterations):
        print(f"\n--- iteration {step + 1} ---")
        messages = trim_history(messages)

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

            print(f"[TOOL RESULT] {output}")
            messages.append({"role": "tool", "content": output})

    print("[STOPPED] Hit max_iterations without a final answer.")
    return None


if __name__ == "__main__":
    # Run 1: teach it something...
    # run_agent(                                    # <-- comment this whole call out
    #     "Remember this: Vaibhav is targeting Agentic AI Engineer roles "
    #     "in India with a compensation goal of 20-40 LPA."
    # )

    # Run 2: start a script with NO mention of the fact above...
    run_agent("What do you remember about my job search goals?")   # <-- uncomment this