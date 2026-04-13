import os
import requests
from functools import lru_cache

# Offloading to HuggingFace Free API - 0MB RAM usage!
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def _get_headers():
    # We will set this in the Render dashboard in Step 3
    api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not api_key:
        print("[WARNING] HUGGINGFACE_API_KEY not found. API calls might fail.")
        return {}
    return {"Authorization": f"Bearer {api_key}"}

def _prepare_text(text: str, is_query: bool = False) -> str:
    text = text.strip()
    if is_query:
        return f"query: {text}"
    return f"passage: {text}"

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    processed = [_prepare_text(t, is_query=False) for t in texts]
    
    response = requests.post(
        API_URL, 
        headers=_get_headers(), 
        json={"inputs": processed, "options": {"wait_for_model": True}}
    )
    
    if response.status_code != 200:
        raise Exception(f"Embedding API Error: {response.text}")
        
    return response.json()

@lru_cache(maxsize=4096)
def _cached_query_embedding(query: str):
    processed = _prepare_text(query, is_query=True)
    response = requests.post(
        API_URL, 
        headers=_get_headers(), 
        json={"inputs": [processed], "options": {"wait_for_model": True}}
    )
    
    if response.status_code != 200:
        raise Exception(f"Embedding API Error: {response.text}")
        
    return tuple(response.json()[0])

def embed_query(query: str) -> list[float]:
    return list(_cached_query_embedding(query))

def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def batch_similarity(query_emb, doc_embs):
    return [cosine_similarity(query_emb, d) for d in doc_embs]