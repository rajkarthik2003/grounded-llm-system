import os

USE_LOCAL = os.getenv("USE_LOCAL_MODEL", "0") == "1"

# ---- Groq (baseline) ----
from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are a medical information assistant. "
    "Use ONLY the provided context. "
    "If the context does not contain the answer, say you cannot answer safely. "
    "Do not add external medical facts."
)

# ---- Local Phi-3 + LoRA (fine-tuned) ----
_local_model = None
_local_tokenizer = None

def _load_local():
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    path = "finetuned_phi3_lora"
    _local_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    _local_model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

def generate_answer(question: str, context: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    if USE_LOCAL:
        _load_local()
        import torch
        inputs = _local_tokenizer(prompt, return_tensors="pt")
        if hasattr(_local_model, "device"):
            inputs = {k: v.to(_local_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _local_model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )
        text = _local_tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split("Answer:")[-1].strip()

    # Groq baseline
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()
