from app.rag.index import retrieve
from app.rag.llm import generate_answer
from app.rag.grounding import check_groundedness

def answer_question(question: str) -> dict:
    docs = retrieve(question)

    # CASE 1: No documents retrieved → refusal (UNGROUNDED)
    if not docs:
        return {
            "answer": "I could not find relevant medical information in the provided documents.",
            "grounded": False,
            "violations": ["No supporting documents retrieved"]
        }

    context = "\n".join(d["text"] for d in docs)
    answer = generate_answer(question, context)

    grounding = check_groundedness(answer, context)

    # CASE 2: Documents exist but answer not supported → refusal (UNGROUNDED)
    if not grounding["grounded"]:
        return {
            "answer": "The available information is insufficient to answer this question safely.",
            "grounded": False,
            "violations": grounding["violations"]
        }

    # CASE 3: Supported answer → grounded
    return {
        "answer": answer,
        "grounded": True,
        "violations": []
    }
