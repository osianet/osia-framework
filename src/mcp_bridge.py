import asyncio
import argparse
import os
import sys
from fastapi import FastAPI, Request
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import uvicorn

# This bridge takes a local STDIO MCP tool and maps it 1:1 to an SSE endpoint.
# This allows AnythingLLM (Docker) to use tools that only exist on the host.

parser = argparse.ArgumentParser(description="OSIA MCP Stdio-to-SSE Bridge")
parser.add_argument("--command", required=True, help="Command to run the STDIO MCP server")
parser.add_argument("--port", type=int, required=True, help="Port to serve SSE on")
parser.add_argument("--name", required=True, help="Internal name for the MCP server")
parser.add_argument("--env-path", help="Optional PYTHONPATH")
parser.add_argument("remainder", nargs=argparse.REMAINDER, help="Arguments for the MCP server")

args = parser.parse_args()
mcp_args = args.remainder
if mcp_args and mcp_args[0] == "--args":
    mcp_args = mcp_args[1:]

# Initialize the MCP Server (The Bridge)
mcp_server = Server(args.name)
sse_transport = SseServerTransport("/messages")

@mcp_server.list_tools()
async def handle_list_tools():
    """Proxies the tool list from the underlying STDIO server."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.list_tools()

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None):
    """Proxies tool calls to the underlying STDIO server."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(name, arguments)

app = FastAPI(title=f"OSIA Bridge: {args.name}")

@app.get("/sse")
async def sse(request: Request):
    async with sse_transport.connect_scope(request.scope, request.receive, request.send):
        await mcp_server.run(
            sse_transport.read_socket,
            sse_transport.write_socket,
            mcp_server.create_initialization_options()
        )

@app.post("/messages")
async def messages(request: Request):
    await sse_transport.handle_post_request(request.scope, request.receive, request.send)

if __name__ == "__main__":
    # Configure the underlying STDIO server parameters
    env = os.environ.copy()
    if args.env_path:
        env["PYTHONPATH"] = args.env_path

    server_params = StdioServerParameters(
        command=args.command,
        args=mcp_args,
        env=env
    )

    print(f"[*] OSIA: Bridging {args.name} (stdio) to SSE on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
