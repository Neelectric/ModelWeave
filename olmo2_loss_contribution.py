from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

model_id = "allenai/OLMo-2-1124-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

dataset = load_dataset("lighteval/MATH")["train"]
instance = dataset[0]
problem = instance["problem"]
chat = [
    {"role":"user", "contet":problem},
    {"role":"assistant", "content":""},
]
templated = tokenizer.apply_chat_template(chat, tokenize=False)
inputs = tokenizer(templated, return_tensors="pt")
outputs = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=500,
)
generation = tokenizer.batch_decode(outputs)
print(generation)