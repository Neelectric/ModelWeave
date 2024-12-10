import re
import shutil
from huggingface_hub import list_repo_refs
from transformers import AutoModelForCausalLM, AutoTokenizer

out = list_repo_refs("allenai/OLMo-2-1124-7B")
branches = [b.name for b in out.branches]

branches_stage1 = [branch for branch in branches if 'stage2' not in branch][1:]
# branches_stage1_healed = [f'stage1-{branch}' if 'stage1' not in branch else branch for branch in branches_stage1]
# for branch in branches_stage1_healed:
#     print(branch)



def extract_step(branch_name):
    match = re.search(r'step(\d+)', branch_name)
    return int(match.group(1)) if match else float('inf')

branches_stage1_healed_sorted = sorted(branches_stage1, key=extract_step)
# for branch in branches_stage1_healed_sorted:
#     print(branch)

# for every branch, i would like to download the model into a cache_dir in this folder, then run the following bash command, and then delete the model from cache dir again

# lm_eval --model hf \
#     --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
#     --tasks lambada_openai,hellaswag \
#     --device cuda:0 \
#     --batch_size 8

for branch in branches_stage1_healed_sorted:
    # download model into cache_dir 
    cache_dir = "OLMo_2_cache/"
    model_name = "allenai/OLMo-2-1124-7B"
    model_revision = branch
    olmo = AutoModelForCausalLM.from_pretrained(
        model_name, 
        revision=model_revision,
        cache_dir=cache_dir,
        )
    command = f"lm_eval --model hf --model_args pretrained={model_name},revision={model_revision},dtype=float --tasks hendrycks_math --batch_size auto"
    print(command)
    # delete model from cache_dir
    del olmo
    # delete folder under "OLMo_2_cache/models--allenai-OLMo-2-1124-7B"
    
    shutil.rmtree(f"{cache_dir}/models--{model_name.replace('/', '-')}-{model_revision}") 
