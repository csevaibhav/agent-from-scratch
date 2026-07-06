"""
MCP Server -- exposes the tools from steps 2/3 (calculator, long-term memory,
file reader) via the actual Model Context Protocol, using Anthropic's
official Python SDK.

The key difference from everything built in steps 1-7: those tools only
ever worked inside YOUR OWN agent loop. This server exposes the same
tools to ANY MCP-compatible client -- Claude Desktop, Claude Code, or
any other MCP client -- without that client knowing anything about your
Python code. The protocol itself defines "here's how to list available
tools" and "here's how to call one," and your job is just to implement
the tool functions -- the SDK handles turning them into that protocol.

Setup:
    pip install mcp

Run directly (for local testing via the MCP Inspector):
    mcp dev mcp_server.py

Or configure it in Claude Desktop's config file to use it for real --
see the README section on MCP for the config snippet.
"""

import ast
import operator
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# The FastMCP object IS the server. Everything below is just registering
# functions to it with a decorator -- similar spirit to how @tool worked
# in the LangGraph comparison, but this time producing a real MCP server
# instead of an in-process LangGraph tool.
# ---------------------------------------------------------------------------
mcp = FastMCP("agent-from-scratch-tools")

# Absolute paths, anchored to this script's own location -- NOT the
# current working directory. This matters because Claude Desktop
# launches this file as a subprocess from its OWN working directory,
# not from inside AGENT-FROM-SCRATCH. A relative path like
# Path("./long_term_memory.json") would silently create/read a
# DIFFERENT file somewhere else when launched that way, causing memory
# to appear to "not work" even though the code runs fine.
SCRIPT_DIR = Path(__file__).resolve().parent
MEMORY_FILE = SCRIPT_DIR / "long_term_memory.json"
SANDBOX_DIR = SCRIPT_DIR / "sandbox_files"
SANDBOX_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# TOOL 1: Calculator -- same safe AST evaluator from steps 2/4/5.
#
# Notice what's DIFFERENT from every previous version: there's no manual
# JSON schema, no TOOLS list, no TOOL_FUNCTIONS dict. The @mcp.tool()
# decorator reads this function's type hints and docstring and generates
# the schema automatically -- then makes it discoverable over MCP to
# whatever client connects.
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
    raise ValueError("Unsupported expression")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, including comparisons
    (e.g. '(12 + 3) * 4' or '3335 > 75')."""
    try:
        return str(_ev(ast.parse(expression, mode="eval").body))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# TOOL 2 & 3: Long-term memory -- same JSON-file store from step 3, now
# reachable by ANY MCP client, not just your own agent script. This means
# Claude Desktop itself could remember facts about you across sessions
# using this exact tool, if you connect this server to it.
# ---------------------------------------------------------------------------
def _load_memory() -> list[str]:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return []


def _save_memory(facts: list[str]) -> None:
    MEMORY_FILE.write_text(json.dumps(facts, indent=2), encoding="utf-8")


@mcp.tool()
def remember(fact: str) -> str:
    """Save an important fact to permanent long-term memory, so it's
    available across future sessions, not just the current conversation."""
    facts = _load_memory()
    facts.append(fact)
    _save_memory(facts)
    return f"Saved to long-term memory: '{fact}'"


@mcp.tool()
def recall_memories() -> str:
    """Retrieve everything saved in long-term memory."""
    facts = _load_memory()
    if not facts:
        return "No long-term memories stored yet."
    return "\n".join(f"- {f}" for f in facts)


# ---------------------------------------------------------------------------
# TOOL 4: File reader -- same sandboxed reader from step 2, path-escape
# protection included, now exposed over MCP.
# ---------------------------------------------------------------------------
@mcp.tool()
def read_sandbox_file(filename: str) -> str:
    """Read the contents of a text file from the local sandbox_files/ folder."""
    try:
        target = (SANDBOX_DIR / filename).resolve()
        if SANDBOX_DIR not in target.parents and target != SANDBOX_DIR:
            return "Error: access outside the sandbox folder is not allowed."
        if not target.exists():
            return f"Error: file '{filename}' not found in sandbox_files/."
        return target.read_text(encoding="utf-8")[:2000]
    except Exception as e:
        return f"Error reading file: {e}"


# ---------------------------------------------------------------------------
# Running this file directly starts the server using STDIO transport --
# the standard way local MCP servers communicate with a client like
# Claude Desktop (the client launches this script as a subprocess and
# talks to it over stdin/stdout, no network port needed).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()