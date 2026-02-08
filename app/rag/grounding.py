def check_groundedness(answer: str, context: str) -> dict:
    """
    Very simple groundedness check:
    - Each sentence in the answer must appear in context
    """
    violations = []

    answer_lines = [
        line.strip() for line in answer.split("\n")
        if line.strip() and not line.lower().startswith("disclaimer")
    ]

    for line in answer_lines:
        if line not in context:
            violations.append(line)

    return {
        "grounded": len(violations) == 0,
        "violations": violations
    }
