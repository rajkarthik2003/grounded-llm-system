from pathlib import Path

# IMPORT INDEX FIRST (CRITICAL)
from app.rag import index
from app.rag.ingest import chunk_text
from app.rag.qa import answer_question

# Load data
text = Path("data/raw/medical_basics.txt").read_text()
chunks = chunk_text(text, "medical_basics")

# Store data
index.store(chunks)

# Ask question
print(answer_question("What is hypertension?"))
