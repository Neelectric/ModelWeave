from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import wandb
from wandb import AlertLevel

manual_seed = 42
torch.random.manual_seed(manual_seed)


### LOAD MODEL, TOKENIZER AND DATASET
# model_id = "allenai/OLMo-2-1124-7B"
model_id = "allenai/OLMo-2-1124-7B-Instruct"
torch_dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch_dtype, # we purposely load in bfloat16 for now to manage storage requirements
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left",
    )
tokenizer.padding_side = "left"
ds_name = "lighteval/MATH"
ds = load_dataset(ds_name, "all")
train = ds["train"]


def format_prompts(instance):
    formatted_prompts = []
    for i in range(len(instance["problem"])):
        chat = [
            {"role": "user", "content": instance['problem'][i]},
            {"role": "assistant", "content": instance['solution'][i]}
            ]
        templated = tokenizer.apply_chat_template(chat, tokenize=False, )
        # return_dict = {
        #     "input_ids": templated,
        #     "attention_mask": torch.ones(len(templated), dtype=torch.bfloat16),
        #     }
        formatted_prompts.append(templated)
    return formatted_prompts

# templated_dataset = train.map(
#     template_func, 
#     )



# def format_prompts(instance):
#     formatted_prompts = []
#     for i in range(len(instance["problem"])):
#         problem = instance["problem"][i]
#         raw_prompt = "Please solve the following problem: " + problem
#         formatted_question = f"<|endoftext|><|user|>\n{raw_prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
#         solution = instance["solution"][i] + EOS_TOKEN
#         prompt = formatted_question + solution
#         formatted_prompts.append(prompt)
#     return formatted_prompts


### TRAINING PREP
# Core training variables and wandb init
bf16 = (torch_dtype == torch.bfloat16)
num_train_epochs = 1
learning_rate = 2e-6
# eval_save_steps = 1000
eval_save_strategy = "epoch"
batch_size = 1
gradient_accumulation_steps = 1

wandb_name = f"MATH_FT - {model_id} on {ds_name}"
print(wandb_name)
wandb.init(
    name= wandb_name,
    project="ModelWeave",
    tags=[model_id],
)
# wandb.alert(
#     title="Starting MATH run",
#     text=f"We have started training",
#     level=AlertLevel.WARN,
#     wait_duration=300,
# )

training_args = SFTConfig(
    # torch_dtype, VRAM and config setup
    fp16= torch_dtype != torch.bfloat16,
    bf16= torch_dtype == torch.bfloat16,

    report_to="wandb",
    # max_steps=max_steps,

    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs={"use_reentrant":False},

    logging_steps=1,
    max_seq_length=32768,
    
    dataset_num_proc=4,
    dataloader_num_workers=4,
    # dataloader_pin_memory=True,
    # dataset_batch_size=batch_size,
    # dataset_batch_size=2500,
    # auto_find_batch_size=True,
    # torch_compile=True,
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
    output_dir="sfttrainer_outputdir",
    run_name= "MATH_FT - " + model_id + ds_name,
    push_to_hub=True,
    hub_model_id="OLMo-2-1124-7B_MATH",
    hub_private_repo=True,
    save_total_limit=1,
    # save_steps=eval_save_steps,
    save_strategy=eval_save_strategy,
    # load_best_model_at_end = True,

    # Eval args
    # eval_strategy=eval_save_strategy,
    # eval_steps=eval_save_steps,
    # per_device_eval_batch_size=batch_size,
    # metric_for_best_model="eval_loss",
)

response_template = "\n<|assistant|>\n"
response_template_tokenized = tokenizer.encode(response_template, add_special_tokens=False,)


trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=train,
    args=training_args,
    formatting_func=format_prompts,
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=response_template_tokenized),
    # data_collator=data_collator,
)
trainer.train()

wandb.alert(
    title="Finished MATH run",
    text=f"We have reached the end of the finetuning script :D",
    level=AlertLevel.WARN,
    wait_duration=300,
)
