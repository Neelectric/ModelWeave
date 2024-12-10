from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import wandb
from wandb import AlertLevel

wandb.init(
    project="ModelWeave",
    tags=["SFT, OLMO-2"],
    )
wandb.alert(
    title="Starting MATH run",
    text=f"We have started training",
    level=AlertLevel.WARN,
    wait_duration=300,
)


### LOAD MODEL, TOKENIZER AND DATASET
model_id = "allenai/OLMo-2-1124-7B"
instruct_model_id = "allenai/OLMo-2-1124-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16, # we purposely load in bfloat16 for now to manage storage requirements
)
tokenizer = AutoTokenizer.from_pretrained(instruct_model_id)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
tokenizer.padding_side = "left"
ds = load_dataset("lighteval/MATH", "all")
print(ds)
train = ds["train"]
print(train)

def template_func(example):
    chat = [
        {"role": "user", "content": example['problem']},
        {"role": "assistant", "content": example['solution']}
        ]
    templated = tokenizer.apply_chat_template(chat, tokenize=True)
    # print(templated)
    return_dict = {
        "input_ids": templated,
        "attention_mask": torch.ones(len(templated), dtype=torch.bfloat16),
        }
    return return_dict

templated_dataset = train.map(
    template_func, 
    # remove_columns=train.column_names,
    # batched=True,
    # num_proc=4,
    )

# train_dataset = tokenizer(templated_dataset, padding=True)

# train_dataloader = DataLoader(
#     train_dataset, 
#     collate_fn=data_collator, 
#     batch_size=16, 
#     shuffle=False,
#     )



### SET UP TRAINING ARGUMENTS
training_args = SFTConfig(
    output_dir="sfttrainer_outputdir",
    report_to="wandb", # enables logging to W&B 😎
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=1,
    
    bf16=True, # use bfloat16 for training
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1, # simulate larger batch sizes

    push_to_hub=True,
    hub_model_id="OLMo-2-1124-7B_MATH",
    hub_private_repo=True,
)


trainer = SFTTrainer(
    model,
    train_dataset=templated_dataset,
    args=training_args,
    # data_collator=data_collator,
)
trainer.train()

wandb.alert(
    title="Finished MATH run",
    text=f"We have reached the end of the finetuning script :D",
    level=AlertLevel.WARN,
    wait_duration=300,
)
