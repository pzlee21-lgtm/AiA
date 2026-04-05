from langchain_chroma import Chroma
from embedding import get_embedding_function
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

def get_vector_store():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

def build_or_load_store():
    """Load Chroma and wrap it so app.py can use .texts and .search()"""
    db = get_vector_store()
    return ChromaWrapper(db)

class ChromaWrapper:
    def __init__(self, db):
        self._db = db
        # This lets app.py do len(STORE.texts) without crashing
        self.texts = db.get()["documents"] or []


    def search(self, query: str, k: int):
        results = self._db.similarity_search_with_score(query, k=k)
        items = []
        rank = 1
        for doc, score in results:
            # if score > 1.5:
            #    print(f"DEBUG: Skipping chunk (distance {score:.3f}) — too dissimilar")
            #    continue
            items.append({
                "rank": rank,
                "text": doc.page_content,
                "meta": doc.metadata,
                "distance": float(score),
            })
            rank += 1
        return items