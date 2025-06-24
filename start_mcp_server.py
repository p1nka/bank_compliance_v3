import asyncio
import logging
from mcp_implementation import MCPServer

logging.basicConfig(level=logging.INFO)


async def main():
    server = MCPServer(host="localhost", port=8765)
    server_instance = await server.start_server()

    print("🚀 MCP Server started on ws://localhost:8765")
    print("Press Ctrl+C to stop...")

    try:
        await server_instance.wait_closed()
    except KeyboardInterrupt:
        print("🛑 Server stopped")


if __name__ == "__main__":
    asyncio.run(main())