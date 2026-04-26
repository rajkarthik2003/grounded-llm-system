# Grounded LLM System

Grounded question-answering system designed to favor evidence-backed responses over fluent but unsupported answers.

## Why This Project Exists

In healthcare and other high-stakes settings, a confident wrong answer is worse than a refusal. This project explores a retrieval-first architecture that treats hallucination as a measurable failure mode and makes grounded behavior the primary goal.

## What The Repository Shows

- Retrieval-augmented QA workflow
- Explicit grounding and refusal-oriented design
- Local ingestion and indexing flow for domain text
- FastAPI-based serving entrypoint
- Lightweight testing path for end-to-end QA behavior

## Current Repository Structure

```text
app/
data/
scripts/
README.md
failure_example.md
requirements.txt
test_qa.py
ui.py
```

## Core Workflow

1. Ingest domain text into chunks.
2. Index chunks into a vector-backed retrieval layer.
3. Retrieve supporting context for a user question.
4. Generate an answer only from retrieved evidence.
5. Refuse or constrain output when support is weak.

## Design Principles

- Retrieval is the only knowledge source used at answer time.
- Hallucination is treated as an evaluation target, not a side effect.
- Refusal can be the correct answer.
- Safety-oriented behavior matters more than fluent completion.

## Repository Evidence

The current codebase publicly shows:

- `FastAPI` in the dependency set
- `chromadb` for local vector storage
- `sentence-transformers` for embeddings
- An ingestion/indexing/testing path centered on `test_qa.py`

This repo is best read as a compact implementation of grounded QA system design rather than a full production deployment artifact.

## Example Local Run

```bash
pip install -r requirements.txt
python test_qa.py
```

## Why It Matters

This project reflects the kind of LLM engineering work I care about most:

- measurable trustworthiness
- retrieval-backed behavior
- failure-aware evaluation
- safe answers in high-risk domains

## Related Experience

My broader resume work around grounded medical RAG includes evaluation-focused iteration, hallucination-rate reduction, and safety-driven refusal behavior. This repository captures the public implementation slice of that direction.
