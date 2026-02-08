from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

ds = load_dataset("json", data_files="data/finetune_instructions.jsonl")["train"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora)

def format_example(ex):
    text = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n{ex['output']}\n"
    )
    tok = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    tok["labels"] = tok["input_ids"].copy()
    return tok

ds = ds.map(format_example)

args = TrainingArguments(
    output_dir="finetuned_phi3_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
)

trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
model.save_pretrained("finetuned_phi3_lora")
tokenizer.save_pretrained("finetuned_phi3_lora")
print("Saved fine-tuned LoRA to finetuned_phi3_lora/")
