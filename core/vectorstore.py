import os
import math
from pinecone import Pinecone, ServerlessSpec
from core.embeddings import embed_texts, embed_query
from functools import lru_cache


class VectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = os.environ.get("PINECONE_INDEX", "docmind-vectors")
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)
        print(f"[NEXUS] Pinecone index '{self.index_name}' ready.")

    def _ensure_index(self):
        existing = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            print(f"[NEXUS] Creating Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("[NEXUS] Index created.")

    @lru_cache(maxsize=2048)
    def _cached_query_embedding(self, query: str):
        return tuple(embed_query(query))

    def _get_query_embedding(self, query: str):
        return [float(x) for x in self._cached_query_embedding(query)]

    def add_documents(self, chunks: list[str], metadata: list[dict], doc_id: str):
        raw_embeddings = embed_texts(chunks)
        embeddings = [[float(x) for x in emb] for emb in raw_embeddings]
        ids = [m["chunk_id"] for m in metadata]

        vectors = []
        for i, chunk_id in enumerate(ids):
            vectors.append({
                "id": chunk_id,
                "values": embeddings[i],
                "metadata": {
                    "doc_id": str(metadata[i]["doc_id"]),
                    "filename": str(metadata[i]["filename"]),
                    "page": int(metadata[i]["page"]),
                    "chunk_index": int(metadata[i]["chunk_index"]),
                    "text": chunks[i][:1000],  # Pinecone metadata limit
                }
            })

        # Batch upsert in groups of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i + batch_size])

    def semantic_search(
        self,
        query: str,
        doc_ids: list[str],
        top_k: int = 10,
        use_mmr: bool = True,
    ) -> list[dict]:
        query_embedding = self._get_query_embedding(query)

        filter_expr = {"doc_id": {"$in": doc_ids}} if doc_ids else None

        results = self.index.query(
            vector=query_embedding,
            top_k=min(top_k * 2, 30),
            filter=filter_expr,
            include_metadata=True,
        )

        seen = set()
        hits = []
        for match in results.matches:
            text = match.metadata.get("text", "")
            key = text[:100]
            if key in seen:
                continue
            seen.add(key)
            hits.append({
                "chunk_id": match.id,
                "text": text,
                "metadata": {
                    "doc_id": match.metadata.get("doc_id"),
                    "filename": match.metadata.get("filename"),
                    "page": int(match.metadata.get("page", 1)),
                    "chunk_index": match.metadata.get("chunk_index", 0),
                },
                "score": round(match.score, 4),
            })

        if use_mmr and len(hits) > top_k:
            hits = self._mmr_filter(hits, top_k)

        return hits[:top_k]

    def _mmr_filter(self, hits: list[dict], top_k: int) -> list[dict]:
        """Simple MMR using score and text diversity."""
        selected = [hits[0]]
        remaining = hits[1:]

        while len(selected) < top_k and remaining:
            best = None
            best_score = -1

            for candidate in remaining:
                relevance = candidate["score"]
                redundancy = max(
                    self._text_similarity(candidate["text"], s["text"])
                    for s in selected
                )
                score = 0.7 * relevance - 0.3 * redundancy
                if score > best_score:
                    best_score = score
                    best = candidate

            if best:
                selected.append(best)
                remaining.remove(best)

        return selected

    def _text_similarity(self, a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def delete_document(self, doc_id: str):
        results = self.index.query(
            vector=[0.0] * 384,
            top_k=1000,
            filter={"doc_id": {"$eq": doc_id}},
            include_metadata=False,
        )
        ids = [m.id for m in results.matches]
        if ids:
            for i in range(0, len(ids), 100):
                self.index.delete(ids=ids[i:i + 100])

    def get_chunk_count(self) -> int:
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count
        except Exception:
            return 0

    def get_all_documents(self) -> dict[str, list]:
        """Fetch all vectors grouped by doc_id for BM25 rebuild."""
        try:
            stats = self.index.describe_index_stats()
            total = stats.total_vector_count
            if total == 0:
                return {}

            # Pinecone doesn't support full scan directly
            # We use a dummy vector query with high top_k per namespace
            results = self.index.query(
                vector=[0.1 for _ in range(384)],
                top_k=min(total, 10000),
                include_metadata=True,
            )

            grouped: dict[str, list] = {}
            for match in results.matches:
                doc_id = match.metadata.get("doc_id")
                if not doc_id:
                    continue
                if doc_id not in grouped:
                    grouped[doc_id] = []
                grouped[doc_id].append({
                    "text": match.metadata.get("text", ""),
                    "metadata": {
                        "doc_id": doc_id,
                        "filename": match.metadata.get("filename", "Unknown"),
                        "page": int(match.metadata.get("page", 1)),
                        "chunk_index": match.metadata.get("chunk_index", 0),
                    }
                })

            return grouped

        except Exception as e:
            print(f"[NEXUS] Could not rebuild from Pinecone: {e}")
            return {}

    def get_stats(self):
        return {"total_chunks": self.get_chunk_count()} 