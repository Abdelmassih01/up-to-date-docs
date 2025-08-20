import os
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from app.models.page import PageDocument

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB  = os.getenv("MONGODB_DB", "docsdb")

async def init_db():
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[MONGODB_DB]
    await init_beanie(database=db, document_models=[PageDocument])
