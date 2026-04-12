from sentence_transformers import SentenceTransformer
from functools import lru_cache
import torch
import os

_model = None


# ---------------------------
# LOAD MODEL (THREAD-SAFE + DEVICE AWARE)
# ---------------------------
def get_model() -> SentenceTransformer:
    global _model

    if _model is None:
        print("[NEXUS] Loading embedding model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        _model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=device,
        )

        # enable faster inference
        _model.eval()

        print(f"[NEXUS] Embedding model ready on {device}.")

    return _model


# ---------------------------
# DYNAMIC BATCH SIZE (⚡ SMART)
# ---------------------------
def _get_batch_size(n: int) -> int:
    if n < 32:
        return 16
    if n < 128:
        return 32
    if n < 512:
        return 64
    return 128


# ---------------------------
# TEXT PREP (OPTIONAL BOOST)
# ---------------------------
def _prepare_text(text: str, is_query: bool = False) -> str:
    text = text.strip()

    # instruction-style embeddings (boosts retrieval quality)
    if is_query:
        return f"query: {text}"
    return f"passage: {text}"


# ---------------------------
# EMBED TEXTS (MAX OPTIMIZED)
# ---------------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = get_model()

    # prepare texts
    processed = [_prepare_text(t, is_query=False) for t in texts]

    batch_size = _get_batch_size(len(processed))

    with torch.no_grad():
        embeddings = model.encode(
            processed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    return embeddings.tolist()


# ---------------------------
# QUERY CACHE (🔥 HUGE SPEED)
# ---------------------------
@lru_cache(maxsize=4096)
def _cached_query_embedding(query: str):
    model = get_model()

    processed = _prepare_text(query, is_query=True)

    with torch.no_grad():
        emb = model.encode(
            processed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    return tuple(emb)


# ---------------------------
# EMBED QUERY (CACHED)
# ---------------------------
def embed_query(query: str) -> list[float]:
    return list(_cached_query_embedding(query))


# ---------------------------
# BULK PARALLEL EMBEDDING (CPU BOOST)
# ---------------------------
def embed_texts_parallel(texts: list[str], num_workers: int = 4):
    """
    Uses multiprocessing for very large ingestion jobs
    """
    from multiprocessing import Pool

    if not texts:
        return []

    chunk_size = max(1, len(texts) // num_workers)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    with Pool(num_workers) as pool:
        results = pool.map(embed_texts, chunks)

    # flatten
    return [emb for sublist in results for emb in sublist]


# ---------------------------
# OPTIONAL: SIMILARITY UTILS
# ---------------------------
def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def batch_similarity(query_emb, doc_embs):
    return [cosine_similarity(query_emb, d) for d in doc_embs]