import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Map of available MCP tools and their startup commands
MCP_SERVERS = {
    "arxiv": StdioServerParameters(
        command="/home/ubuntu/mcp-arxiv/.venv/bin/python",
        args=["-m", "arxiv_mcp_server"],
        env={"PYTHONPATH": "/home/ubuntu/mcp-arxiv/src", **os.environ}
    ),
    "semantic-scholar": StdioServerParameters(
        command="/home/ubuntu/mcp-semantic-scholar/.venv/bin/python",
        args=["-m", "semantic_scholar_mcp.cli", "serve"],
        env={"PYTHONPATH": "/home/ubuntu/mcp-semantic-scholar/src", **os.environ}
    ),
    "wikipedia": StdioServerParameters(
        command="/home/ubuntu/mcp-wikipedia-py/.venv/bin/python",
        args=["/home/ubuntu/mcp-wikipedia-py/main.py"],
        env=os.environ
    ),
    "tavily": StdioServerParameters(
        command="npx",
        args=["-y", "tavily-mcp"],
        env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""), **os.environ}
    )
}

class MCPDispatcher:
    """Dispatches calls to various MCP servers to gather external intelligence."""

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        if server_name not in MCP_SERVERS:
            raise ValueError(f"Unknown MCP server: {server_name}")

        server_params = MCP_SERVERS[server_name]
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print(f"[*] MCP: Calling {server_name} -> {tool_name} with {arguments}")
                result = await session.call_tool(tool_name, arguments)
                return result
