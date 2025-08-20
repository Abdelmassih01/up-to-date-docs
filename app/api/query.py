from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Optional
import chromadb

# connect to chroma (point to your container port)
chroma_client = chromadb.HttpClient(host="localhost", port=8001)
collection = chroma_client.get_or_create_collection("crawled_docs_pages")

router = APIRouter(prefix="/query", tags=["Query"])


class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5


class QueryResponse(BaseModel):
    ids: List[str]
    documents: List[str]
    metadatas: List[dict]


@router.post("/", response_model=QueryResponse)
def query_collection(body: QueryRequest):
    results = collection.query(
        query_texts=[body.query],
        n_results=body.n_results
    )
    return QueryResponse(
        ids=results["ids"][0],
        documents=results["documents"][0],
        metadatas=results["metadatas"][0],
    )
