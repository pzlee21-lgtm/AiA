"""
Two-stage hybrid reranker for SmartPharma RAG.
Stage 1: BM25 keyword scoring
Stage 2: Cross-Encoder neural scoring
"""
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

# ================================================
# Load Cross-Encoder model once when app starts
# This avoids reloading it on every single request
# ================================================
print("Loading Cross-Encoder reranker model...")
_CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-Encoder ready.")


def bm25_score(query: str, items: List[Dict[str, Any]]) -> List[float]:
    """
    Score documents using BM25 keyword matching.
    Good at matching exact drug names and dosages.
    """
    tokenized_docs = [item["text"].lower().split() for item in items]
    tokenized_query = query.lower().split()
    bm25 = BM25Okapi(tokenized_docs)
    return bm25.get_scores(tokenized_query).tolist()


def cross_encoder_score(query: str, items: List[Dict[str, Any]]) -> List[float]:
    """
    Score documents using Cross-Encoder neural model.
    Good at understanding clinical context and meaning.
    """
    pairs = [[query, item["text"]] for item in items]
    return _CROSS_ENCODER.predict(pairs).tolist()


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to the range 0-1.
    """
    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        return [1.0 for _ in scores]

    return [(s - min_s) / (max_s - min_s) for s in scores]


def hybrid_rerank(
    query: str,
    items: List[Dict[str, Any]],
    top_k: int = 5,
    bm25_weight: float = 0.4,
    cross_encoder_weight: float = 0.6,
    min_score: float = 0.4       #filters out low relevance docs
) -> List[Dict[str, Any]]:
    """
    Two-stage hybrid reranker combining BM25 + Cross-Encoder.

    Args:
        query: The user's question
        items: Retrieved documents from Chroma
        top_k: How many results to return after reranking
        bm25_weight: Weight for keyword score (default 30%)
        cross_encoder_weight: Weight for neural score (default 70%)
        min_score: Minimum combined score to include a document (default 0.3)

    Returns:
        Reranked list of top_k documents above min_score threshold
    """
    # Safety check — if no documents, return empty
    if not items:
        print("[WARN] Reranker received empty document list")
        return items

    print(f"DEBUG: Reranking {len(items)} documents...")

    # BM25 Scoring
    bm25_scores = bm25_score(query, items)
    print(f"DEBUG: BM25 scores: {[round(s, 3) for s in bm25_scores]}")

    # Cross-Encoder Scoring
    ce_scores = cross_encoder_score(query, items)
    print(f"DEBUG: Cross-Encoder scores: {[round(s, 3) for s in ce_scores]}")

    # Normalize both score lists to 0-1
    bm25_norm = normalize_scores(bm25_scores)
    ce_norm = normalize_scores(ce_scores)

    # Combine scores using weights
    combined = []
    for i, item in enumerate(items):
        final_score = (
            (bm25_weight * bm25_norm[i]) +
            (cross_encoder_weight * ce_norm[i])
        )
        combined.append((final_score, item))
        print(f"DEBUG: Doc {i+1} — BM25: {round(bm25_norm[i], 3)}, "
              f"CE: {round(ce_norm[i], 3)}, "
              f"Final: {round(final_score, 3)}")

    # Sort highest score first
    combined.sort(key=lambda x: x[0], reverse=True)

    # ✅ CHANGED — filter out docs below min_score before returning
    reranked = []
    actual_rank = 1  # ✅ separate counter that only increments for kept docs
    for new_rank, (score, item) in enumerate(combined[:top_k]):
        if score < min_score:
            print(f"DEBUG: Dropping doc {new_rank + 1} "
                f"(score {score:.3f}) — below threshold {min_score}")
            continue
        item["rank"] = actual_rank  # ✅ always starts from 1 consecutively
        actual_rank += 1
        reranked.append(item)

    print(f"DEBUG: Reranking complete. Returning {len(reranked)} documents.")
    return reranked