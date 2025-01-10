from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from tqdm import tqdm

### LOAD MODEL, TOKENIZER, DATASET
model_id = "CohereForAI/c4ai-command-r7b-12-2024"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    # attn_implementation="eager", # need to do this because "Olmo2Model is using Olmo2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, need to specify this"
)
dataset = load_dataset("lighteval/MATH", trust_remote_code=True)["test"]
output_dir = "min_heatmaps"
raw_heatmap_dir = os.path.join(output_dir, "raw_heatmaps")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(raw_heatmap_dir, exist_ok=True)

num_instances = len(dataset)
# num_instances = 5
global_min = float('inf')
global_max = float('-inf')

cross_dataset_mins = []
### PREPARE AND TOKENIZE FIRST PROMPT
for instance_idx in tqdm(range(num_instances), dynamic_ncols=True):
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
        # output_attentions=True,
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
    tqdm.write(f"Loss: {loss.item()}")
    loss.backward()

    ### CALCULATE MEAN GRADIENT PER LAYER
    instance_mins = []
    for layer_id, layer in enumerate(model.model.layers):
        layer_mins = []
        # first we grab gradients from the 4 self_attn matrices
        self_attn = layer.self_attn
        self_attn_matrices = [self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj]
        for matrix in self_attn_matrices:
            grad = matrix.weight.grad
            min_grad = grad.min()
            layer_mins.append(min_grad)
        
        # now try the same for 3 mlp matrices
        mlp = layer.mlp
        mlp_matrices = [mlp.gate_proj, mlp.up_proj, mlp.down_proj]
        for matrix in mlp_matrices:
            grad = matrix.weight.grad
            min_grad = grad.min()
            layer_mins.append(min_grad)
        instance_mins.append(layer_mins)
    model.zero_grad()
    # min, max and curtosis!
    # could be interesting if gradient is skewed --> only updating certain neurons and others less
    # plot matrices individually
    # full ft is never low rank
    # can we do lora? plot delta
    # merge model wtih adapter then plot before and after
    # maybe ideally gradient should update sparse set of neurons instead of everything
    # maybe a loss function that enforces sparsity in weight updates?
    # maybe we don't need sparsity but skewedness is enough
    # get lora, then take delta W adn it may be more sparse and low rank than full finetuning


    # separate math loss skewedness from correct/iincorrect predictions
    # only do gradients if they exceed a certain threshold!
    # research methods of skewedness, curtosis and Pearson mode skewness and Pearson median skewness
    # check out skewedness of residual stream!
    # gradients similar to saliency maps!
    # this can be a preliminary study for some actual hardened mechinterp research


    # take top 100 neurons where gradient is highest
    # then knock them out
    # what is impact? does accuracy go up or down
    # if it goes up, then gradient was pinpointing most incorrect neurons
    # feels like pruning? gradient based pruning?
    # borrow ideas from lottery ticket hypothesis work?
    # 3 datasets: math, entity tracking and natural lagnauge
    # neurons that light up on entity tracking should 
    # revisit monosemanticity and SAEs!

    # Convert the list of lists to a tensor
    instance_mins_tensor = torch.tensor(instance_mins).to(torch.float32) # need to convert to fp32 otherwise plt complains
    cross_dataset_mins.append(instance_mins_tensor)

    # Update global min and max values
    global_min = min(global_min, instance_mins_tensor.min().item())
    global_max = max(global_max, instance_mins_tensor.max().item())

# # Plot the heatmap for each instance with consistent color scale
# for instance_idx, instance_mins_tensor in tqdm(enumerate(cross_dataset_mins), dynamic_ncols=True):
#     plt.figure(figsize=(12, 8))
#     plt.title(f"Min of Gradients Heatmap of {model_id} on MATH - Instance {instance_idx + 1}")
#     plt.imshow(instance_mins_tensor, cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
#     plt.colorbar(label='Min Gradient')
#     plt.xlabel("Attention and MLP Matrices")
#     plt.ylabel("Layers")
#     plt.gca().invert_yaxis()  # Invert y-axis to start layers at 0 at the bottom
#     plt.xticks(ticks=range(7), labels=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
#     plt.yticks(ticks=range(32), labels=range(32))
#     plt.savefig(os.path.join(raw_heatmap_dir, f"heatmap_{instance_idx + 1}.png"))
#     plt.close()

# # Create a GIF from the saved heatmap images
# images = []
# for instance_idx in range(num_instances):
#     image_path = os.path.join(raw_heatmap_dir, f"heatmap_{instance_idx + 1}.png")
#     images.append(imageio.imread(image_path))

# gif_path = os.path.join(output_dir, f"min_gradients_heatmaps_{num_instances}.gif")
# imageio.mimsave(gif_path, images, duration=1)  # Increase duration to 2 seconds per frame

# print(f"GIF saved at {gif_path}")

cross_dataset_mins_tensor = torch.stack(cross_dataset_mins)
averaged_dataset_mins = cross_dataset_mins_tensor.min(dim=0).values
plt.figure(figsize=(12, 8))
plt.title(f"Min Gradients Heatmap of {model_id} on MATH  - Average of {num_instances}")
plt.imshow(averaged_dataset_mins, cmap='viridis', aspect='auto')
plt.colorbar(label='Min Gradient')
plt.xlabel("Attention and MLP Matrices")
plt.ylabel("Layers")
plt.gca().invert_yaxis()  # Invert y-axis to start layers at 0 at the bottom
plt.xticks(ticks=range(7), labels=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
plt.yticks(ticks=range(32), labels=range(32))
plt.savefig(os.path.join(output_dir, f"min_of_{num_instances}_heatmap_.png"))
plt.close()