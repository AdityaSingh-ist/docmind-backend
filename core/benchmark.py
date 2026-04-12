import os
import time
import asyncio
import statistics
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TEST_CASES = [
    {"id": f"q{i+1}", "question": q, "category": c}
    for i, (q, c) in enumerate([
        ("What is the main objective or purpose described in this document?", "comprehension"),
        ("What methodology or approach is used?", "methodology"),
        ("What are the key findings or conclusions?", "findings"),
        ("What data sources or datasets are referenced?", "data"),
        ("What limitations or challenges are mentioned?", "limitations"),
        ("What recommendations or future work is suggested?", "recommendations"),
        ("Who are the key stakeholders or target audience mentioned?", "stakeholders"),
        ("What metrics or performance indicators are used?", "metrics"),
        ("What technologies, tools, or frameworks are mentioned?", "technology"),
        ("What is the timeframe or timeline discussed?", "timeline"),
    ])
]


def score_retrieval(chunks: List[Dict]) -> Dict:
    if not chunks:
        return {"score": 0, "coverage": 0, "diversity": 0, "chunk_count": 0}

    pages = set(c.get("page", 0) for c in chunks)
    docs = set(c.get("doc_id", "") for c in chunks)

    coverage = min(len(pages) / 4.0, 1.0)
    diversity = min(len(docs) / 3.0, 1.0)

    scores = [c.get("final_score", c.get("rerank_score", c.get("score", 0))) for c in chunks]
    normalized = [1 / (1 + abs(s - 0.5)) for s in scores]
    rank_quality = sum(normalized) / len(normalized) if normalized else 0

    overall = 0.4 * coverage + 0.2 * diversity + 0.4 * rank_quality

    return {
        "score": round(overall, 3),
        "coverage": round(coverage, 3),
        "diversity": round(diversity, 3),
        "chunk_count": len(chunks),
        "pages_hit": sorted(list(pages)),
    }


def score_answer_quality(answer: str) -> Dict:
    if not answer or len(answer.strip()) < 20:
        return {"score": 0, "word_count": 0}

    words = answer.split()
    wc = len(words)

    citation_hits = sum(1 for x in ["page", "document", "source", "according"] if x in answer.lower())
    citation_score = min(citation_hits / 2, 1.0)

    has_structure = any(x in answer for x in ["**", "##", "- ", "\n\n"])
    length_score = min(wc / 150, 1.0)

    score = 0.4 * citation_score + 0.35 * length_score + 0.25 * (1 if has_structure else 0)

    return {
        "score": round(max(score, 0), 3),
        "word_count": wc,
        "citations": citation_hits,
    }


async def run_single_case(case, retriever, doc_ids, doc_registry, stream_response_fn, provider):
    question = case["question"]
    start = time.time()

    try:
        retrieved = retriever.retrieve(question, doc_ids, top_k=6)
    except Exception as e:
        return {"id": case["id"], "category": case["category"], "question": question, "status": "retrieval_failed", "error": str(e)}

    retrieval_scores = score_retrieval(retrieved)

    context = "\n\n".join([
        f"[{doc_registry.get(c['doc_id'], {}).get('filename', 'Unknown')}, Page {c['page']}]\n{c['text']}"
        for c in retrieved
    ])

    answer = ""
    try:
        async for token in stream_response_fn(query=question, context=context, provider=provider, history=[]):
            answer += token
    except Exception as e:
        answer = f"[LLM Error: {e}]"

    latency = round((time.time() - start) * 1000, 1)
    answer_scores = score_answer_quality(answer)

    return {
        "id": case["id"],
        "category": case["category"],
        "question": question,
        "status": "success",
        "retrieval": retrieval_scores,
        "answer_scores": answer_scores,
        "latency_ms": latency,
        "answer_preview": answer[:400] + "..." if len(answer) > 400 else answer,
        "chunks_retrieved": retrieval_scores["chunk_count"],
        "pages_hit": retrieval_scores.get("pages_hit", []),
    }


async def run_benchmark(
    retriever,
    doc_ids: List[str],
    doc_registry: Dict,
    test_cases: Optional[List[Dict]] = None,
    llm_provider: str = "groq",
) -> Dict:
    from core.llm import stream_response

    if not test_cases:
        test_cases = DEFAULT_TEST_CASES

    # Sequential to avoid rate limits on free tier
    results = []
    for tc in test_cases:
        result = await run_single_case(tc, retriever, doc_ids, doc_registry, stream_response, llm_provider)
        results.append(result)
        await asyncio.sleep(0.5)  # rate limit buffer

    successes = [r for r in results if r.get("status") == "success"]
    ns = len(successes)

    retrieval_scores = [r["retrieval"]["score"] for r in successes]
    answer_scores = [r["answer_scores"]["score"] for r in successes]
    latencies = [r["latency_ms"] for r in successes]

    summary = {
        "total_questions": len(results),
        "successful": ns,
        "failed": len(results) - ns,
        "avg_retrieval_score": round(sum(retrieval_scores) / ns, 3) if ns else 0,
        "avg_quality_score": round(sum(answer_scores) / ns, 3) if ns else 0,
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "overall_score": round((sum(retrieval_scores) + sum(answer_scores)) / (2 * ns), 3) if ns else 0,
    }

    return {
        "summary": summary,
        "results": results,
        "llm_provider": llm_provider,
        "doc_ids_tested": doc_ids,
    }