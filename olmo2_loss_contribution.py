from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt

### LOAD MODEL, TOKENIZER, DATASET
model_id = "allenai/OLMo-2-1124-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # torch_dtype=torch.bfloat16,
    attn_implementation="eager", # need to do this because "Olmo2Model is using Olmo2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, need to specify this"
)
dataset = load_dataset("lighteval/MATH", trust_remote_code=True)["test"]

### PREPARE AND TOKENIZE FIRST PROMPT
instance = dataset[0]
problem = instance["problem"]
solution = instance["solution"]
print(solution)
chat = [
    {"role": "user", "content": problem},
]
templated = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(templated, return_tensors="pt").to(model.device)
input_ids = inputs["input_ids"]

### CALL .GENERATE() AND PRINT OUTPUT
# outputs = model.generate(
#     input_ids,
#     labels=input_ids,
#     do_sample=False,
#     max_new_tokens=1,
#     # output_hidden_states=True,
#     # output_attentions=True,
#     # output_loss=True
#     # output_scores=True,
#     # return_dict_in_generate=True,
# )
# print(outputs[-1])

### FORWARD PASS, PRINT PREDICTION
for param in model.parameters():
    param.requires_grad = True
outputs = model(
    **inputs,
    labels=inputs["input_ids"],
    return_dict_in_generate=True,
    output_attentions=True,
)

# loss_tuple=outputs.loss,
logits=outputs.logits[0],
past_key_values=outputs.past_key_values
hidden_states=outputs.hidden_states
attentions=outputs.attentions


next_token_logits = outputs.logits[:, -1, :].clone().float()
next_token_logits = next_token_logits.to(input_ids.device)

next_tokens = torch.argmax(next_token_logits, dim=-1)
print(next_tokens)
input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

generation = tokenizer.batch_decode(input_ids)
print(generation)

# loss = loss_tuple[0]
### BACKWARD PASS
for att in attentions:
    att.requires_grad_(True)
target = input_ids[:, 1:].contiguous()
logits = outputs.logits[:, :-1, :].contiguous()

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(outputs.logits.view(-1, model.config.vocab_size), target.view(-1))
loss.backward()

# Extract attention gradients
attention_gradients = [att.grad for att in attentions]

# Check if gradients are None
for i, grad in enumerate(attention_gradients):
    if grad is None:
        print(f"Gradients for attention layer {i} are None")

# Calculate contributions for each attention head to the final token
# Multiplying attention weights by their gradients
attention_contributions = [att[:, :, -1, :] * grad[:, :, -1, :] for att, grad in zip(attentions, attention_gradients) if grad is not None]
print(len(attention_gradients))
print(attention_gradients[0].shape)
attention_contributions = attention_gradients[:, :, -1, :]

# Aggregate and normalize contributions
attention_scores = torch.stack(attention_contributions).squeeze()
attention_scores = attention_scores[:,:,-1] # convert from [num_layers,num_heads,num_tokens] to [32,32,last_token]

# Visualization
plt.figure(figsize=(12, 6))
plt.title("Attention Head Contributions to Final Token")
plt.imshow(attention_scores.cpu().detach().numpy(), cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel("Attention Heads")
plt.ylabel("Layers")
plt.gca().invert_yaxis() 
plt.show()