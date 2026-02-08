from pathlib import Path
from app.rag.ingest import chunk_text
from app.rag.vectorstore import index_chunks

print("STARTING LOAD")
text = Path("data/raw/medical_basics.txt").read_text()
chunks = chunk_text(text, "medical_basics")
index_chunks(chunks)
print(f"INDEXED {len(chunks)} chunks")
