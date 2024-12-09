from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_conf = BitsAndBytesConfig(load_in_4bit=True)

model_id = "allenai/OLMo-2-1124-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # bnb_config=bnb_conf,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
