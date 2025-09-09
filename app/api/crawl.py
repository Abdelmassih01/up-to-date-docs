from fastapi import APIRouter
from pydantic import BaseModel, AnyHttpUrl
from app.services.crawler_service import crawl_site
import asyncio

router = APIRouter()

class CrawlRequest(BaseModel):
    base_url: AnyHttpUrl

@router.post("")
async def start_crawl(request: CrawlRequest):
    asyncio.create_task(crawl_site(str(request.base_url)))
    return {"message": f"Crawl started for {request.base_url}"}
