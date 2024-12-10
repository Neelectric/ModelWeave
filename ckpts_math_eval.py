import re
import shutil
import csv
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




# for branch in branches_stage1_healed_sorted:
#     cache_dir = "OLMo_2_cache/"
#     model_name = "allenai/OLMo-2-1124-7B"
#     model_revision = branch
#     olmo = AutoModelForCausalLM.from_pretrained(
#         model_name, 
#         revision=model_revision,
#         cache_dir=cache_dir,
#         )
#     command = f"lm_eval --model hf --model_args pretrained={model_name},revision={model_revision},dtype=float --tasks hendrycks_math --batch_size auto"
#     print(command)
#     del olmo
#     shutil.rmtree(f"{cache_dir}/models--{model_name.replace('/', '-')}-{model_revision}") 



def execute_command_and_get_performance(command):
    # Implement the function to execute the command and return the performance result
    # For example, you can use subprocess to run the command and parse the output
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Parse the result.stdout to extract the performance metric
    performance = parse_performance(result.stdout)
    return performance

def parse_performance(output):
    # Implement the function to parse the performance from the command output
    # This is a placeholder implementation
    performance = output.split('Performance: ')[1].split('\n')[0]
    return performance
# Open the CSV file in append mode
with open('performance_results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header if the file is empty
    if file.tell() == 0:
        writer.writerow(['Branch', 'Performance'])

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
        del olmo
        command = f"lm_eval --model hf --model_args parallelize=True,pretrained={model_name},revision={model_revision},cache_dir={cache_dir},dtype=bfloat16 --tasks hendrycks_math --batch_size 16 --output_path results"
        print(command)
        
        # Execute the command and capture the performance result
        performance_result = execute_command_and_get_performance(command)
        
        # Write the performance result to the CSV file
        writer.writerow([branch, performance_result])
        
        
        # delete folder under "OLMo_2_cache/models--allenai-OLMo-2-1124-7B"
        shutil.rmtree(f"{cache_dir}/models--{model_name.replace('/', '-')}-{model_revision}")

# lm_eval --model hf --model_args parallelize=True,pretrained=sfttrainer_outputdir/,dtype=bfloat16 --tasks hendrycks_math --batch_size 16 --output_path results