import math
from rank_bm25 import BM25Okapi
from core.vectorstore import VectorStore

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            print("[NEXUS] Loading reranker...")
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("[NEXUS] Reranker ready.")
        except Exception as e:
            print(f"[NEXUS] Reranker unavailable: {e}")
            return None
    return _reranker


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def reciprocal_rank_fusion(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            items[cid] = item

    fused = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [{"score": scores[cid], **items[cid]} for cid in fused]


class HybridRetriever:
    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore
        self.bm25_indices: dict[str, tuple] = {}

    def index_bm25(self, doc_id: str, chunks: list[str]):
        tokenized = [c.lower().split() for c in chunks]
        self.bm25_indices[doc_id] = (BM25Okapi(tokenized), chunks)

    def remove_bm25(self, doc_id: str):
        self.bm25_indices.pop(doc_id, None)

    def bm25_search(self, query: str, doc_ids: list[str], top_k: int = 10) -> list[dict]:
        query_tokens = query.lower().split()
        all_hits = []

        for doc_id in doc_ids:
            if doc_id not in self.bm25_indices:
                continue
            bm25, chunks = self.bm25_indices[doc_id]
            raw_scores = bm25.get_scores(query_tokens)

            for i, score in enumerate(raw_scores):
                if score > 0:
                    all_hits.append({
                        "chunk_id": f"{doc_id}_bm25_{i}",
                        "text": chunks[i],
                        "doc_id": doc_id,
                        "score": float(score),
                        "page": 1,
                    })

        all_hits.sort(key=lambda x: x["score"], reverse=True)
        return all_hits[:top_k]

    def retrieve(self, query: str, doc_ids: list[str], top_k: int = 6) -> list[dict]:
        # Stage 1 — semantic search
        semantic_hits = self.vectorstore.semantic_search(query, doc_ids, top_k=10)
        semantic_normalized = [
            {
                "chunk_id": h["chunk_id"],
                "text": h["text"],
                "doc_id": h["metadata"]["doc_id"],
                "page": h["metadata"]["page"],
                "score": h["score"],
            }
            for h in semantic_hits
        ]

        # Stage 2 — BM25 search
        bm25_hits = self.bm25_search(query, doc_ids, top_k=10)

        # Build page map for enriching BM25 results
        page_map = {h["chunk_id"]: h["page"] for h in semantic_normalized}

        # Stage 3 — RRF fusion
        fused = reciprocal_rank_fusion([semantic_normalized, bm25_hits])

        for item in fused:
            if item.get("page", 1) == 1:
                item["page"] = page_map.get(item["chunk_id"], 1)

        # Stage 4 — Cross-encoder reranking (lazy, None-safe)
        if len(fused) > 1:
            try:
                reranker = get_reranker()
                if reranker is not None:
                    pairs = [[query, item["text"]] for item in fused]
                    raw_scores = reranker.predict(pairs)
                    for i, item in enumerate(fused):
                        item["rerank_score"] = round(sigmoid(float(raw_scores[i])), 4)
                        item["final_score"] = item["rerank_score"]
                    fused.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
                else:
                    for item in fused:
                        item["rerank_score"] = item.get("score", 0)
                        item["final_score"] = item["rerank_score"]
            except Exception as e:
                print(f"[NEXUS] Reranker fallback: {e}")
                for item in fused:
                    item["rerank_score"] = item.get("score", 0)
                    item["final_score"] = item["rerank_score"]

        return fused[:top_k]

    def retrieve_fast(self, query: str, doc_ids: list[str], top_k: int = 4) -> list[dict]:
        hits = self.vectorstore.semantic_search(query, doc_ids, top_k=top_k)
        return [
            {
                "chunk_id": h["chunk_id"],
                "text": h["text"],
                "doc_id": h["metadata"]["doc_id"],
                "page": h["metadata"]["page"],
                "score": h["score"],
                "rerank_score": h["score"],
                "final_score": h["score"],
            }
            for h in hits
        ]