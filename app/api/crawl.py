from fastapi import APIRouter
from pydantic import BaseModel, AnyHttpUrl
from app.services.crawler_service import crawl_site

router = APIRouter()

class CrawlRequest(BaseModel):
    base_url: AnyHttpUrl

@router.post("/crawl")
async def start_crawl(request: CrawlRequest):
    # For now: run inline (simple MVP). You can move to BackgroundTasks later.
    await crawl_site(str(request.base_url))
    return {"message": f"Crawl finished for {request.base_url}"}
