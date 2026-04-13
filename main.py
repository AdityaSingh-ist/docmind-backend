import os
import uuid
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from core.ingestion import ingest_pdf
from core.vectorstore import VectorStore
from core.retrieval import HybridRetriever
from core.llm import stream_response
from core.benchmark import run_benchmark

load_dotenv()

app = FastAPI(title="NEXUS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Your local frontend
        "https://docmind-frontend-flame.vercel.app" # Your deployed Vercel frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def rebuild_state_from_vectorstore():
    """Rebuild BM25 indices and doc_registry from persisted ChromaDB on startup."""
    grouped = vectorstore.get_all_documents()
    if not grouped:
        return

    print(f"[NEXUS] Rebuilding state for {len(grouped)} documents from ChromaDB...")

    for doc_id, chunks in grouped.items():
        texts = [c["text"] for c in chunks]
        meta = chunks[0]["metadata"]
        filename = meta.get("filename", "Unknown")
        max_page = max(c["metadata"].get("page", 1) for c in chunks)

        retriever.index_bm25(doc_id, texts)

        upload_path = UPLOAD_DIR / f"{doc_id}.pdf"
        size_kb = round(upload_path.stat().st_size / 1024, 1) if upload_path.exists() else 0

        doc_registry[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": len(texts),
            "pages": max_page,
            "size_kb": size_kb,
        }

    print(f"[NEXUS] State rebuilt. {len(doc_registry)} docs ready.")


try:
    vectorstore = VectorStore()
    retriever = HybridRetriever(vectorstore)
    doc_registry: dict[str, dict] = {}
    rebuild_state_from_vectorstore()
    print("[NEXUS] Startup complete.")
except Exception as e:
    print(f"[NEXUS] Startup error: {e}")
    # Still define these so the app doesn't crash on import
    vectorstore = None
    retriever = None
    doc_registry = {}


# ─── Models ───────────────────────────────────────────────────────────────────

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    doc_ids: list[str] | None = None
    llm_provider: str = "groq"
    conversation_history: list = []
    mode: str = "fast"   # 👈 properly indented


class BenchmarkRequest(BaseModel):
    doc_ids: list[str] | None = None
    llm_provider: str = "groq"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "docs_indexed": len(doc_registry),
        "chunks_in_store": vectorstore.get_chunk_count(),
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{doc_id}.pdf"

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        chunks, metadata = ingest_pdf(str(file_path), doc_id, file.filename)
        vectorstore.add_documents(chunks, metadata, doc_id)
        retriever.index_bm25(doc_id, chunks)

        doc_registry[doc_id] = {
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_count": len(chunks),
            "pages": metadata[-1]["page"] if metadata else 0,
            "size_kb": round(len(content) / 1024, 1),
        }

        return doc_registry[doc_id]

    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@app.get("/documents")
def list_documents():
    return list(doc_registry.values())


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if doc_id not in doc_registry:
        raise HTTPException(404, "Document not found.")

    vectorstore.delete_document(doc_id)
    retriever.remove_bm25(doc_id)

    file_path = UPLOAD_DIR / f"{doc_id}.pdf"
    file_path.unlink(missing_ok=True)
    del doc_registry[doc_id]

    return {"status": "deleted", "doc_id": doc_id}


@app.post("/query")
async def query_documents(req: QueryRequest):
    if not doc_registry:
        raise HTTPException(400, "No documents indexed. Upload documents first.")

    target_ids = req.doc_ids if req.doc_ids else list(doc_registry.keys())
    mode = req.mode if hasattr(req, "mode") else "fast"

    if mode == "fast":
        retrieved_chunks = retriever.retrieve_fast(req.query, target_ids, top_k=4)

    elif mode == "balanced":
        retrieved_chunks = retriever.retrieve_balanced(req.query, target_ids, top_k=5)

    else:
        retrieved_chunks = retriever.retrieve(req.query, target_ids, top_k=6)

    if not retrieved_chunks:
        raise HTTPException(404, "No relevant content found.")

    async def event_stream() -> AsyncGenerator[str, None]:
        citations = [
            {
                "doc_id": c["doc_id"],
                "filename": doc_registry.get(c["doc_id"], {}).get("filename", "Unknown"),
                "page": c["page"],
                "text": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                "score": round(c.get("rerank_score", c.get("score", 0)), 4),
            }
            for c in retrieved_chunks
        ]
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

        context = "\n\n---\n\n".join([
            f"[Source: {doc_registry.get(c['doc_id'], {}).get('filename', 'Unknown')}, Page {c['page']}]\n{c['text']}"
            for c in retrieved_chunks
        ])

        async for token in stream_response(
            query=req.query,
            context=context,
            provider=req.llm_provider,
            history=req.conversation_history,
        ):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest):
    if not doc_registry:
        raise HTTPException(400, "No documents indexed. Upload documents first.")

    target_ids = req.doc_ids if req.doc_ids else list(doc_registry.keys())

    try:
        results = await run_benchmark(
            retriever=retriever,
            doc_ids=target_ids,
            doc_registry=doc_registry,
            llm_provider=req.llm_provider,
        )
        return results
    except Exception as e:
        raise HTTPException(500, f"Benchmark failed: {str(e)}")