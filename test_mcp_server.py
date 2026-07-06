"""
Quick sanity test for mcp_server.py -- calls each tool directly through
the MCP interface, without needing the Node.js-based Inspector UI.

Run:
    python test_mcp_server.py
"""

import asyncio
import mcp_server as server


async def main():
    print("Registered tools:")
    tools = await server.mcp.list_tools()
    for t in tools:
        print(f"  - {t.name}: {t.description[:70]}")

    print("\n--- Testing calculator ---")
    result = await server.mcp.call_tool("calculator", {"expression": "145 * 23"})
    print(result)

    print("\n--- Testing remember ---")
    result = await server.mcp.call_tool("remember", {"fact": "Testing the MCP server directly."})
    print(result)

    print("\n--- Testing recall_memories ---")
    result = await server.mcp.call_tool("recall_memories", {})
    print(result)

    print("\n--- Testing read_sandbox_file ---")
    result = await server.mcp.call_tool("read_sandbox_file", {"filename": "notes.txt"})
    print(result)


if __name__ == "__main__":
    asyncio.run(main())