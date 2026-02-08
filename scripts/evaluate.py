import json
from pathlib import Path

from app.rag.ingest import chunk_text
from app.rag import index
from app.rag.qa import answer_question

# ---- LOAD DATA (CRITICAL) ----
text = Path("data/raw/medical_basics.txt").read_text()
chunks = chunk_text(text, "medical_basics")
index.store(chunks)

print(f"[INFO] Loaded {len(chunks)} chunks for evaluation")

# ---- LOAD EVAL QUESTIONS ----
eval_data = json.loads(Path("data/eval_questions.json").read_text())

# ---- RUN EVALUATION ----
total = 0
grounded_true = 0
grounded_false = 0

for item in eval_data:
    total += 1
    print("=" * 60)
    print("Question:", item["question"])
    print("Expected:", item["expected"])

    result = answer_question(item["question"])

    print("Model Answer:", result["answer"])
    print("Grounded:", result["grounded"])

    if result["grounded"]:
        grounded_true += 1
    else:
        grounded_false += 1
        print("Violations:", result["violations"])

print("\n" + "#" * 60)
print(f"TOTAL: {total}")
print(f"GROUNDED_TRUE: {grounded_true}")
print(f"GROUNDED_FALSE: {grounded_false}")
print(f"HALLUCINATION_RATE: {grounded_false/total:.2f}")
print("#" * 60)

