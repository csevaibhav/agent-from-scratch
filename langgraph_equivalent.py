"""
LangGraph equivalent of step1_ollama.py / step2_three_tools.py -- the
basic ReAct loop + tool dispatch, rebuilt with the framework.

Compare this to your step2_three_tools.py side by side. Notice what's
GONE (you don't write it) vs what's IDENTICAL (you still write it,
just plugged into framework slots).

Setup:
    pip install langgraph langchain-ollama

Run:
    python langgraph_equivalent.py
"""

from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ---------------------------------------------------------------------------
# TOOLS -- this part is IDENTICAL work to what you already did. LangGraph
# doesn't know how to do arithmetic any better than raw Ollama does; you
# still write the actual logic. The only change is a decorator instead of
# a manual JSON schema dict -- @tool auto-generates the schema from your
# function's type hints and docstring.
# ---------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '(12 + 3) * 4'."""
    import ast, operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


tools = [calculator]

# ---------------------------------------------------------------------------
# THIS is what LangGraph actually replaces: your hand-written dispatch
# table (TOOL_FUNCTIONS dict) AND the manual "loop over tool_calls, run
# each, append tool_result" block from your run_agent(). ToolNode does
# both, generically, for any list of @tool-decorated functions.
# ---------------------------------------------------------------------------
tool_node = ToolNode(tools)

llm = ChatOllama(model="llama3.1").bind_tools(tools)


# ---------------------------------------------------------------------------
# STATE -- this is your `messages` list from step 1-3, formalized as a
# typed schema instead of a bare Python list. `add_messages` is a
# reducer -- it tells LangGraph "when a node returns new messages,
# APPEND them to existing state, don't overwrite it." You wrote this
# append-don't-overwrite behavior by hand every time you did
# `messages.append(...)`.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# THE "REASON" NODE -- this is the part of your loop that calls the model.
# You still write this. LangGraph doesn't decide what to send the model;
# you do, same as step 1.
# ---------------------------------------------------------------------------
def call_model(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# THE GRAPH -- this replaces your `for step in range(max_iterations):`
# while-loop. Instead of a Python loop with an if-check on stop_reason,
# you declare NODES (units of work) and EDGES (what runs next), and
# `tools_condition` is a prebuilt function that does the exact same
# check you wrote by hand: "did the last message request a tool call?
# if so go to the tool node, otherwise end."
# ---------------------------------------------------------------------------
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    tools_condition,      # same logic as: `if response.stop_reason == "tool_use"`
    {"tools": "tools", END: END},
)
graph.add_edge("tools", "agent")  # after running tools, loop back to reasoning

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke(
        {"messages": [{"role": "user", "content": "What is (145 * 23) + 7, then divide that by 3?"}]}
    )
    print(result["messages"][-1].content)