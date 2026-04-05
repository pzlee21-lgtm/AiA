# """"
# Retrieval and ranking logic for SmartPharma RAG.
# Handles section filtering and source-based re-ranking of search results.
# """

# from typing import List, Dict, Any
# from vector_store import get_vector_store


# def filter_by_section(
#     items: List[Dict[str, Any]], 
#     allowed_sources: set[str]
# ) -> List[Dict[str, Any]]:
#     """
#     Filter retrieved items to only those from allowed sources.
    
#     Args:
#         items: List of retrieved document dictionaries
#         allowed_sources: Set of allowed source identifiers
        
#     Returns:
#         Filtered list of items
#     """
#     return [
#         item for item in items 
#         if item.get("meta", {}).get("source") in allowed_sources
#     ]


# def rank_with_source_bonus(
#     items: List[Dict[str, Any]], 
#     bonus_sources: set[str], 
#     bonus: float = 0.15
# ) -> List[Dict[str, Any]]:
#     """
#     Re-rank items by giving bonus (lower distance) to preferred sources.
    
#     Args:
#         items: List of retrieved document dictionaries
#         bonus_sources: Set of preferred source identifiers
#         bonus: Distance reduction for preferred sources (default 0.15)
        
#     Returns:
#         Re-ranked list of items sorted by adjusted distance
#     """
#     ranked = []
    
#     for item in items:
#         score = item["distance"]
        
#         # Apply bonus if from preferred source
#         if item.get("meta", {}).get("source") in bonus_sources:
#             score -= bonus
            
#         ranked.append((score, item))
    
#     # Sort by adjusted score
#     ranked.sort(key=lambda x: x[0])
    
#     return [item for _, item in ranked]


# def retrieve_context(query: str, k: int = 5):
#     db = get_vector_store()

#     results = db.similarity_search_with_score(query, k=k)

#     items = []
#     for doc, score in results:   # ✅ unpack properly
#         items.append({
#             "text": doc.page_content,
#             "meta": doc.metadata,
#             "distance": score
#         })

#     return items