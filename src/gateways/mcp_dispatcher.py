import asyncio
import logging
import os
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("osia.mcp")


def _build_server_configs() -> dict[str, StdioServerParameters]:
    """Build MCP server configs from env vars so paths aren't hardcoded."""
    mcp_base = os.getenv("MCP_TOOLS_BASE", "/home/ubuntu")
    osia_base = os.getenv("OSIA_BASE_DIR", "/home/ubuntu/osia-framework")

    return {
        "arxiv": StdioServerParameters(
            command=f"{mcp_base}/mcp-arxiv/.venv/bin/python",
            args=["-m", "arxiv_mcp_server"],
            env={"PYTHONPATH": f"{mcp_base}/mcp-arxiv/src", **os.environ},
        ),
        "semantic-scholar": StdioServerParameters(
            command=f"{mcp_base}/mcp-semantic-scholar/.venv/bin/python",
            args=["-m", "semantic_scholar_mcp.cli", "serve"],
            env={"PYTHONPATH": f"{mcp_base}/mcp-semantic-scholar/src", **os.environ},
        ),
        "wikipedia": StdioServerParameters(
            command=f"{mcp_base}/mcp-wikipedia-py/.venv/bin/python",
            args=[f"{mcp_base}/mcp-wikipedia-py/main.py"],
            env=os.environ,
        ),
        "tavily": StdioServerParameters(
            command="npx",
            args=["-y", "tavily-mcp"],
            env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""), **os.environ},
        ),
        "youtube": StdioServerParameters(
            command="node",
            args=[
                f"{mcp_base}/osia-knowledge-base/mcp/youtube/"
                "node_modules/@fabriqa.ai/youtube-transcript-mcp/index.js"
            ],
            env=os.environ,
        ),
        "time": StdioServerParameters(
            command="uvx",
            args=["mcp-server-time"],
            env=os.environ,
        ),
    }


class MCPDispatcher:
    """Dispatches calls to various MCP servers with session caching."""

    def __init__(self):
        self._configs = _build_server_configs()
        self._sessions: dict[str, tuple] = {}  # name -> (session, cleanup)
        self._locks: dict[str, asyncio.Lock] = {}

    async def _get_session(self, server_name: str) -> ClientSession:
        """Return a cached session, creating one if needed."""
        if server_name not in self._locks:
            self._locks[server_name] = asyncio.Lock()

        async with self._locks[server_name]:
            if server_name in self._sessions:
                return self._sessions[server_name][0]

            server_params = self._configs[server_name]
            # Enter the context managers manually so we can keep them alive
            transport_ctx = stdio_client(server_params)
            read, write = await transport_ctx.__aenter__()
            session_ctx = ClientSession(read, write)
            session = await session_ctx.__aenter__()
            await session.initialize()

            # Store both so we can clean up later
            self._sessions[server_name] = (session, session_ctx, transport_ctx)
            logger.info("Opened persistent MCP session for %s", server_name)
            return session

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        if server_name not in self._configs:
            raise ValueError(f"Unknown MCP server: {server_name}")

        try:
            session = await self._get_session(server_name)
            logger.info("Calling %s -> %s with %s", server_name, tool_name, arguments)
            return await session.call_tool(tool_name, arguments)
        except Exception as e:
            logger.warning("MCP call to %s/%s failed: %s — recycling session", server_name, tool_name, e)
            await self._close_session(server_name)
            # Retry once with a fresh session
            session = await self._get_session(server_name)
            return await session.call_tool(tool_name, arguments)

    async def _close_session(self, server_name: str):
        entry = self._sessions.pop(server_name, None)
        if entry:
            session, session_ctx, transport_ctx = entry
            try:
                await session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await transport_ctx.__aexit__(None, None, None)
            except Exception:
                pass

    async def close_all(self):
        for name in list(self._sessions):
            await self._close_session(name)
