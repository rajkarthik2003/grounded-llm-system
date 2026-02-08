Grounded LLM System with Evaluation-Driven Alignment
Overview

This project implements a production-style LLM system that prioritizes correctness and safety over fluent but unsupported answers. The system uses retrieval-augmented generation (RAG), explicit grounding checks, and refusal logic to prevent hallucinations in high-risk scenarios.

System Architecture
User Query
   ↓
Semantic Retriever (Vector DB)
   ↓
Context Assembler
   ↓
Grounding & Safety Gate
   ├── Refuse (if unsupported)
   └── LLM (API or Local Fine-tuned)
            ↓
        Final Answer
            ↓
     Offline Evaluation Loop

Key Design Choices

Retrieval is the only source of knowledge

Hallucination is treated as a measurable failure

Refusal is considered a correct outcome

Fine-tuning is applied only after evaluation

Evaluation

Evaluation was performed on a manually curated set of question–answer pairs designed to test grounding and refusal behavior.

Metric tracked: hallucination rate
Baseline → tuned improvement: 1.00 → 0.33

Tradeoffs

The system intentionally favors refusal over speculative answers, accepting lower coverage in exchange for higher trustworthiness.

Production Relevance

This mirrors real-world LLM constraints in enterprise and high-risk domains where correctness matters more than fluency.