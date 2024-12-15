from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
import imageio
import os

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
output_dir = "heatmaps"
os.makedirs(output_dir, exist_ok=True)
num_instances = 10

cross_dataset_means = []
### PREPARE AND TOKENIZE FIRST PROMPT
for instance_idx in range(num_instances):
    instance = dataset[instance_idx]
    problem = instance["problem"]
    solution = instance["solution"]
    chat = [
        {"role": "user", "content": problem},
    ]
    templated = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=True)
    inputs = tokenizer(templated, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    ### FORWARD PASS AND NEXT TOKEN PREDICTION
    for param in model.parameters():
        param.requires_grad = True
    outputs = model(
        **inputs,
        labels=inputs["input_ids"],
        return_dict_in_generate=True,
        output_attentions=True,
    )
    logits=outputs.logits[0],
    next_token_logits = outputs.logits[:, -1, :].clone().float()
    next_token_logits = next_token_logits.to(input_ids.device)

    next_tokens = torch.argmax(next_token_logits, dim=-1)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    generation = tokenizer.batch_decode(input_ids)

    ### CALCULATE LOSS AND BACKWARD PASS
    target = input_ids[:, 1:].contiguous()
    logits = outputs.logits[:, :-1, :].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(outputs.logits.view(-1, model.config.vocab_size), target.view(-1))
    print(f"Loss: {loss.item()}")
    loss.backward()

    ### CALCULATE MEAN GRADIENT PER LAYER
    all_means = []
    print("WE ARE IGNORING FUCKING OLMO RMSNORM FOR EVERY ATTN PARAM")
    for layer_id, layer in enumerate(model.model.layers):
        layer_means = []
        # first we grab gradients from the 4 self_attn matrices
        self_attn = layer.self_attn
        self_attn_matrices = [self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj]
        for matrix in self_attn_matrices:
            grad = matrix.weight.grad
            mean_grad = grad.mean()
            layer_means.append(mean_grad)
        
        # now try the same for 3 mlp matrices
        mlp = layer.mlp
        mlp_matrices = [mlp.gate_proj, mlp.up_proj, mlp.down_proj]
        for matrix in mlp_matrices:
            grad = matrix.weight.grad
            mean_grad = grad.mean()
            layer_means.append(mean_grad)
        all_means.append(layer_means)

    # Convert the list of lists to a tensor
    all_means_tensor = torch.tensor(all_means)
    cross_dataset_means.append(all_means_tensor)

# print(len(cross_dataset_means))
# cross_dataset_means_tensor = torch.stack(cross_dataset_means)
# averaged_dataset_means = cross_dataset_means_tensor.mean(dim=0)
# print(averaged_dataset_means.shape)

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    plt.title(f"Mean Gradients Heatmap - Instance {instance_idx + 1}")
    plt.imshow(all_means_tensor, cmap='viridis', aspect='auto') # change this to averaged_dataset_means
    plt.colorbar(label='Mean Gradient')
    plt.xlabel("Attention and MLP Matrices")
    plt.ylabel("Layers")
    plt.gca().invert_yaxis()  # Invert y-axis to start layers at 0 at the bottom
    plt.xticks(ticks=range(7), labels=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    plt.yticks(ticks=range(32), labels=range(32))
    # plt.show()
    plt.savefig(os.path.join(output_dir, f"heatmap_{instance_idx + 1}.png"))
    plt.close()

# Create a GIF from the saved heatmap images
images = []
for instance_idx in range(num_instances):
    image_path = os.path.join(output_dir, f"heatmap_{instance_idx + 1}.png")
    images.append(imageio.imread(image_path))

gif_path = os.path.join(output_dir, "mean_gradients_heatmaps.gif")
imageio.mimsave(gif_path, images, duration=2)

print(f"GIF saved at {gif_path}")