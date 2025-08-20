from fastapi import FastAPI
from app.core.db import init_db
from app.core.vector import init_vector           # <-- add this
from app.api.crawl import router as crawl_router
from app.api.public_pages import router as public_router
from app.api.query import router as query_router

app = FastAPI(title="Docs Crawler API")

app.include_router(crawl_router, prefix="/api", tags=["crawl"])
app.include_router(public_router)
app.include_router(query_router, prefix="/api")

@app.on_event("startup")
async def on_startup():
    await init_db()
    init_vector()

