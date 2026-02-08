import uuid
import re

def chunk_text(text, source):
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    for i, sentence in enumerate(sentences):
        if len(sentence.strip()) < 20:
            continue

        chunks.append({
            "id": str(uuid.uuid4()),
            "text": sentence.strip(),
            "metadata": {
                "source": source,
                "chunk_id": f"{source}_{i}"
            }
        })

    return chunks
