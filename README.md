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
| `api.py` | Deployment | Wraps the agent as a real HTTP service using FastAPI — a request comes in over `/agent`, the agent plans + executes, and structured JSON comes back, instead of running a script and reading a terminal |
| `mcp_server.py` + `test_mcp_server.py` | Standardized tool access | Exposes the calculator, memory, and file-reader tools via the real [Model Context Protocol](https://modelcontextprotocol.io/) using Anthropic's official Python SDK — tested standalone, then connected to and verified working in Claude Desktop |

## Setup

**1. Install [Ollama](https://ollama.com/download) and pull a tool-calling-capable model:**
```bash
ollama pull llama3.1
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
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
- **Relative paths break when a different process launches your script** — `mcp_server.py` originally used `Path("./long_term_memory.json")`. This worked fine when running the file directly, but silently failed when Claude Desktop launched it as a subprocess from its own working directory — it was reading/writing a *different* file location entirely, with no error, just apparently "not remembering" anything. Fixed by anchoring paths to the script's own location (`Path(__file__).resolve().parent`) instead of the current working directory.
- **Tool selection ambiguity between overlapping capabilities** — after connecting the MCP server to Claude Desktop, asking it to "remember" something was answered by Claude's own *native* memory feature instead of the connected `remember` MCP tool, since both could plausibly satisfy the request. Confirmed by checking the actual `long_term_memory.json` file on disk (unchanged) versus Claude's response (which referenced its native memory, not the tool). Resolved by explicitly naming both the tool and the server in the prompt, removing the ambiguity.

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

## Running as an API

`api.py` wraps the same planning + async execution logic from steps 4–7 behind a real HTTP interface using [FastAPI](https://fastapi.tiangolo.com/), so the agent can be called by another program instead of run manually as a script.

**Install the extra dependencies:**
```bash
pip install fastapi uvicorn
```

**Start the server:**
```bash
uvicorn api:app --reload
```
`--reload` restarts the server automatically on code changes — useful during development, drop it for a production run.

**Try it interactively:** open `http://127.0.0.1:8000/docs` in a browser. FastAPI auto-generates a full interactive interface from the code — no separate tool needed to test requests.

**Or call it directly:**
```bash
curl -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d "{\"request\": \"Calculate 145 * 23, then calculate 900 / 12\"}"
```

**Endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/agent` | POST | Submit a request; returns the plan's subtask breakdown, per-subtask status, and results as structured JSON |
| `/health` | GET | Basic liveness check |
| `/logs/recent` | GET | Returns the most recent structured log entries (same data as `analyze_logs.py`, retrievable over HTTP) |

Request/response shapes are validated automatically via Pydantic models — malformed requests are rejected with a clear error before any agent logic runs, rather than failing partway through.

## MCP Server — connecting these tools to a real product

Every tool built in steps 2–3 only ever worked inside this project's own agent loop. `mcp_server.py` exposes the same calculator, memory, and file-reader tools via the [Model Context Protocol](https://modelcontextprotocol.io/) — an open standard for how AI applications discover and call external tools — using Anthropic's official Python SDK. Once running, **any** MCP-compatible client (Claude Desktop, Claude Code, or any other MCP client) can use these tools directly, without knowing anything about the Python behind them.

**Install:**
```bash
pip install "mcp[cli]"
```

**Test standalone first** (no client needed):
```bash
python test_mcp_server.py
```
This calls each tool directly through the real MCP interface and prints the results — confirms the server itself works before connecting anything to it.

**Optional — visual debugging with the MCP Inspector** (requires [Node.js](https://nodejs.org)):
```bash
mcp dev mcp_server.py
```
Opens a browser UI listing all registered tools, with a form to test each one interactively.

**Connect it to Claude Desktop:**

1. Locate (or create) `%APPDATA%\Claude\claude_desktop_config.json`
2. Add an entry pointing at this server, using absolute paths:
```json
{
  "mcpServers": {
    "agent-from-scratch-tools": {
      "command": "C:\\path\\to\\venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\agent-from-scratch\\mcp_server.py"]
    }
  }
}
```
3. **Fully quit** Claude Desktop (not just close the window) and reopen it
4. Check **"+" → Connectors** in the chat input — `agent-from-scratch-tools` should appear with its 4 tools listed

**Verifying it's actually being used** (not just connected) turned out to be its own small investigation — Claude Desktop has its own native memory feature that can satisfy a vague "remember this" request without ever touching a connected MCP tool. The only conclusive test was writing a fact through an explicit tool call and checking `long_term_memory.json` directly on disk afterward — confirmed working end-to-end this way. See "Real bugs found while building this" above for the two issues this surfaced.

## Stack

Python 3.12 · Ollama (`llama3.1`, local) · `ddgs` (web search) · `asyncio` · FastAPI + Uvicorn (API layer) · MCP (Model Context Protocol, official Python SDK) · LangGraph (comparison only)

## Author

Built by Vaibhav as part of a hands-on transition into Agentic AI / LLM Engineering.