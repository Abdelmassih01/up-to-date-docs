from fastapi import APIRouter, Request
import aiohttp
from mcp.server.fastmcp import FastMCP

# Create the MCP instance
mcp = FastMCP("my-query-api")

@mcp.tool()
async def query(query: str) -> str:
    """
    Query documents from Abdelmassih VPS API.
    """
    url = f"http://abdelmassih.vps.webdock.cloud/query?query={query}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

# FastAPI router wrapper
router = APIRouter()

@router.post("/mcp")
async def handle_mcp(request: Request):
    return await mcp.fastapi_handler(request)
