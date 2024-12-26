### Data collator work

# System imports
import sys
import os

# Local imports
from src.wandb_eval_callback import WandbPredictionProgressCallback

# External imports
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from tqdm import tqdm
import torch
from datasets import Dataset
import wandb

### SYSTEM PREP
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
manual_seed = 42
torch.random.manual_seed(manual_seed)
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

### SETUP TOKENIZER, MODEL AND DATASET
# Load dataset
ds_name = "lighteval/MATH"
train = load_dataset(ds_name, keep_in_memory=True, split="train")
eval = load_dataset(ds_name, keep_in_memory=True, split="test")
def format_prompts(instance):
    formatted_prompts = []
    for i in range(len(instance["problem"])):
        problem = instance["problem"][i]
        raw_prompt = "Please solve the following problem: " + problem
        formatted_question = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n {raw_prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        solution = instance["solution"][i] + EOS_TOKEN
        prompt = formatted_question + solution
        formatted_prompts.append(prompt)
    return formatted_prompts

# Load tokenizer and model
resolve_model_name = {
    "OLMo-2_it": "allenai/OLMo-2-7B-1124"
}
model_name = "OLMo-2_it"
model_id = resolve_model_name[model_name]
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
BOS_TOKEN = tokenizer.bos_token
EOS_TOKEN = tokenizer.eos_token
# tokenizer.pad_token = tokenizer.unk_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
torch_dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch_dtype,
    attn_implementation="flash_attention_2",
    use_cache=False,
)
model.resize_token_embeddings(len(tokenizer)) # can't tell if we need this? may get CUDA device side asserts triggered otherwise which are hard to debug
# model.config.use_cache=False


### TRAINING PREP
# Core training variables and wandb init
bf16 = (torch_dtype == torch.bfloat16)
num_train_epochs = 1
learning_rate = 1e-4
eval_save_steps = 1000
eval_save_strategy = "steps"
batch_size = 4
gradient_accumulation_steps = 1
wandb.init(
    name= f"MATH_FT - {model_name}_{ds_name}",
    project="ModelWeave",
    tags=[model_name],
)

# Set up training arguments and trainer
sft_config = SFTConfig(
    # torch_dtype, VRAM and config setup
    fp16=not bf16,
    bf16=bf16,

    report_to="wandb",
    # max_steps=max_steps,

    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    logging_steps=1,
    max_seq_length=32768,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs={"use_reentrant":False},
    dataset_num_proc=4,
    dataloader_num_workers=4,
    # dataloader_pin_memory=True,
    # dataset_batch_size=batch_size,
    # dataset_batch_size=2500,
    # auto_find_batch_size=True,
    torch_compile=True,
    # torch_compile_mode="max-autotune",

    # LR and optimizer management
    warmup_steps=0,
    warmup_ratio=0.01,
    learning_rate=learning_rate,
    optim="adamw_torch_fused",
    weight_decay=0,
    lr_scheduler_type="linear",
    # lr_scheduler_kwargs={"num_cycles": num_train_epochs},
    seed=manual_seed,
    data_seed=manual_seed,
    max_grad_norm=1.0,

    # Saving args
    output_dir=f"output_dir/{model_name}" ,
    run_name= "MATH_FT - " + model_name + ds_name,
    push_to_hub=True,
    hub_private_repo = True,
    save_total_limit=2,
    # save_steps=eval_save_steps,
    save_strategy=eval_save_strategy,
    load_best_model_at_end = True,

    # Eval args
    eval_strategy=eval_save_strategy,
    # eval_steps=eval_save_steps,
    per_device_eval_batch_size=batch_size,
    metric_for_best_model="eval_loss",
)

response_template = "<|end_header_id|>\n\n"
response_template_tokenized = tokenizer.encode(response_template, add_special_tokens=False,)

trainer = SFTTrainer(
    model,
    train_dataset=train,
    formatting_func=format_prompts,
    eval_dataset=eval,
    tokenizer=tokenizer,
    args=sft_config,
    # data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=response_template_tokenized),
)

# Instantiate WandbPredictionProgressCallback
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    model_id=sft_config.output_dir,
    benchmark="lighteval/MATH",
    num_questions=1190,
    batch_size=2,
)

# Add the callback to the trainer
add_callback = True
if add_callback:
    trainer.add_callback(progress_callback)
    print("Callback added to trainer")
else:
    print("Callback not added to trainer")
trainer_stats = trainer.train()
print(trainer_stats)
trainer.push_to_hub()
wandb.finish()