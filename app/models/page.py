from __future__ import annotations
from datetime import datetime
from typing import List, Optional, Literal
from beanie import Document, Indexed
from pydantic import BaseModel, Field

class CodeBlock(BaseModel):
    language: Optional[str] = None
    code: str

class Section(BaseModel):
    heading: Optional[str] = None
    heading_level: Optional[int] = None
    text: Optional[str] = None
    codes: List[CodeBlock] = Field(default_factory=list)

class Heading(BaseModel):
    level: int
    text: str

class PageDocument(Document):
    url: Indexed(str, unique=True)  # unique index on URL
    title: str
    headings: List[Heading] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    last_crawled: datetime
    # optional fields for future use
    code_blocks_flat: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    hash: Optional[str] = None
    
    embedding_status: Optional[str] = None     # "ok" | "failed" | "stale" | None
    last_embedded_at: Optional[datetime] = None
    embedding_model: Optional[str] = None
    vector_store_id: Optional[str] = None

    class Settings:
        name = "pages"  # collection name
