from typing import List, Dict
import re

DOCUMENTS: List[Dict] = []

def store(chunks: List[Dict]):
    global DOCUMENTS
    DOCUMENTS = chunks

def score(doc_text: str, query: str) -> int:
    words = re.findall(r"\w+", query.lower())
    text = doc_text.lower()
    return sum(1 for w in words if w in text)

def retrieve(query: str, top_k: int = 3) -> List[Dict]:
    scored = []
    for d in DOCUMENTS:
        s = score(d["text"], query)
        if s > 0:
            scored.append((s, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]
