import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("med_docs")

model = SentenceTransformer("all-MiniLM-L6-v2")

def index_chunks(chunks):
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    embeddings = model.encode(texts).tolist()
    collection.add(documents=texts, ids=ids, embeddings=embeddings)

def search(query, k=3):
    q_emb = model.encode([query]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    return res["documents"][0] if res["documents"] else []
