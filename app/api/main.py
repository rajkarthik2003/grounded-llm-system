from fastapi import FastAPI
from pydantic import BaseModel
from app.rag.qa import answer_question
from pathlib import Path
from app.rag.ingest import chunk_text
from app.rag.index import store


app = FastAPI(title="Grounded LLM System")

class AskRequest(BaseModel):
    question: str

@app.on_event("startup")
def load_documents():
    text = Path("data/raw/medical_basics.txt").read_text()
    chunks = chunk_text(text, source="medical_basics")
    store(chunks)
    print(f"[Startup] Loaded {len(chunks)} documents into vector store")

@app.get("/")
def root():
    return {"message": "API running"}

@app.post("/ask")
def ask(req: AskRequest):
    result = answer_question(req.question)
    return {
        "answer": result["answer"],
        "grounded": result["grounded"],
        # keep violations for debugging/demo; remove later if you want
        "violations": result.get("violations", [])
    }
