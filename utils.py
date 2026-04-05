# """
# Utility functions for SmartPharma RAG backend.
# Provides file I/O, directory management, and data loading utilities.
# """

# import os
# import json
# from typing import List, Dict, Any


# def load_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
#     """
#     Load and parse multiple JSONL files.
    
#     Args:
#         paths: List of file paths to JSONL files
        
#     Returns:
#         List of dictionaries containing 'text' and 'meta' fields
#     """
#     docs: List[Dict[str, Any]] = []
    
#     for path in paths:
#         if not os.path.exists(path):
#             print(f"[WARN] Missing source file: {path}")
#             continue
            
#         with open(path, encoding="utf-8") as f:
#             for line in f:
#                 try:
#                     obj = json.loads(line)
#                 except json.JSONDecodeError:
#                     continue
                    
#                 text = (obj.get("text") or "").strip()
#                 if not text:
#                     continue
                    
#                 meta = obj.get("meta") or {}
#                 docs.append({"text": text, "meta": meta})
                
#     return docs


# def ensure_directory(dir_path: str) -> None:
#     """
#     Create directory if it doesn't exist.
    
#     Args:
#         dir_path: Path to directory to create
#     """
#     os.makedirs(dir_path, exist_ok=True)