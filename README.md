# Agent From Scratch

A ReAct-style AI agent built entirely from raw LLM API calls — no LangChain, no CrewAI, no AutoGPT. Built as a hands-on learning project while transitioning into Agentic AI / LLM Engineering, to deeply understand what agent frameworks actually automate before relying on them.

Runs 100% locally and free using [Ollama](https://ollama.com) (`llama3.1`) on a local GPU — no API keys, no cost.

## Why build this without a framework?

Frameworks like LangGraph are genuinely useful, but they hide the mechanics of how an agent actually works: the reasoning loop, tool dispatch, memory, planning, and failure handling. This project builds each piece by hand first, so the framework version (included at the end) can be understood as "what this automates" rather than "magic."

## What's inside

| File | Concept | What it adds |
|---|---|---|
| `step1_ollama.py` | Core reasoning loop | Minimal ReAct loop (Reason → Act → Observe) with one tool |
| `step2_three_tools.py` | Tool dispatch | 3 tools (calculator, web search, file reader) via a manual dispatch dict; safe AST-based math evaluation instead of `eval()` |
| `step3_memory.py` | Memory | Short-term conversation history + persistent long-term fact store (JSON file), exposed as tools the model chooses to use |
| `step4_planning.py` | Planning | Breaks a multi-step request into an explicit subtask list before executing, with context threaded between dependent subtasks |
| `step5_guardrails.py` | Error handling | Timeouts on LLM/tool calls, and failure classification (`transient` vs `permanent`) so retries only happen when they can actually help |
| `step6_async.py` | Async execution | Independent subtasks run concurrently via `asyncio.gather()`; dependent subtasks wait in correct order (dependency-wave execution) |
| `step7_observability.py` + `analyze_logs.py` | Observability | Every event (tool call, retry, failure, subtask/session timing) logged as structured JSON Lines; a separate script queries the logs for stats across runs |
| `langgraph_equivalent.py` | Framework comparison | The same reasoning loop rebuilt in LangGraph, for a direct side-by-side |

## Setup

**1. Install [Ollama](https://ollama.com/download) and pull a tool-calling-capable model:**
```bash
ollama pull llama3.1
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install ollama ddgs
```

**3. Run any step:**
```bash
python step1_ollama.py
```

**Optional — LangGraph comparison:**
```bash
python -m pip install langgraph langchain-ollama
python langgraph_equivalent.py
```

## Real bugs found while building this

Building from scratch surfaces failure modes a framework tutorial would never show you. A few worth documenting:

- **Unsafe `eval()`** — the first version of the calculator used raw `eval()` on model-generated strings. Replaced with an AST-based evaluator that only permits arithmetic, closing an obvious injection risk before it touched anything real.
- **Subtask context isolation** — early planning code ran each subtask in a completely fresh conversation. A "compare the two results" subtask had no idea what the two results *were*, and hallucinated a plausible-looking fake comparison. Fixed by threading prior subtask results forward into later prompts.
- **Retrying failures that can't be fixed by retrying** — a subtask involving an unsupported operation (a division-by-zero style capability gap) got retried 3 times, failing identically each time. Fixed by classifying failures as `transient` (worth retrying) vs `permanent` (retrying is pointless — the tool categorically can't do it), so guardrails now fail fast on the latter.
- **Model synthesis errors despite correct tool results** — on more than one run, the model's final answer contradicted a tool result that was correct and present in its own context (e.g. claiming a file "wasn't provided" when its content had just been read successfully). A reminder that tool execution succeeding does not guarantee the final synthesized answer is correct — always worth verifying both independently.

## What LangGraph replaces (and what it doesn't)

After building the loop by hand, `langgraph_equivalent.py` rebuilds the same reasoning + tool-calling loop using the framework, to see exactly what's automated:

| Hand-written | LangGraph equivalent |
|---|---|
| Manual dispatch dict (`TOOL_FUNCTIONS`) | `@tool` decorator + `ToolNode` |
| Manually written JSON tool schemas | Auto-generated from function signature + docstring |
| `messages.append(...)` everywhere | `Annotated[list, add_messages]` reducer |
| `if stop_reason != "tool_use": return` | `tools_condition` |
| `for step in range(max_iterations):` loop | Graph nodes + edges |

**Not replaced by any framework:** the actual tool logic, the prompts, and the retry/planning/failure-classification decisions from steps 4–6 — those are still yours to design either way. Frameworks provide the plumbing; the reasoning about failure modes and task decomposition is the actual engineering work.

## Stack

Python 3.12 · Ollama (`llama3.1`, local) · `ddgs` (web search) · `asyncio` · LangGraph (comparison only)

## Author

Built by Vaibhav as part of a hands-on transition into Agentic AI / LLM Engineering.