from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from datasets import load_dataset


### LOAD MODEL, TOKENIZER AND DATASET
model_id = "allenai/OLMo-2-1124-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16, # we purposely load in bfloat16 for now to manage storage requirements
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
ds = load_dataset("lighteval/MATH", "all")
print(ds)
train = ds["train"]
print(train)
print(train[0])
first_sample = train[0]

for key, value in first_sample.items():
    print(f"{key}: {value}")
    print("="*100)

# tokenizer.apply_chat_template