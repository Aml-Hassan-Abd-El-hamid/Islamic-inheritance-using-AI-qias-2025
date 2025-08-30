# Unsloth Local Training Script
from unsloth import FastLanguageModel
import torch
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
from unsloth import is_bfloat16_supported
is_bfloat16_supported()
from unsloth.chat_templates import train_on_responses_only
import os
from datasets import load_dataset, concatenate_datasets

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

from datasets import load_dataset

# Load the dataset from CSV
all_fatawas = load_dataset(
    "csv",
    data_files={"train": "Hirtage_Pipeline_Finetune/unsloth/data/all_fatawas.csv"},
    split="train"
)

# Split into train and dev with 10% for dev using a fixed seed
split_dataset = all_fatawas.train_test_split(test_size=0.2, seed=42)

# Now you have both train and dev splits
fatawas_train = split_dataset['train']
fatawas_dev = split_dataset['test']

# Print dataset info
print("Train dataset size:", len(fatawas_train))
print("Dev dataset size:", len(fatawas_dev))


# Configuration
model_name = "unsloth/Qwen3-4B"  # You can change to any other supported model
max_seq_length = 2048
dtype = torch.bfloat16
load_in_4bit = False

# Load base model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = ""
)


# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",

                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)


# fatawas_ prompt formatting
islam_prompt = """المقالة
### الموضوع: {}

### السؤال:{}
<think>
### الإجابة:{}
</think>
"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(example):
    Category = example["Category"]
    Question = example["Question"]
    Answer = example["Answer"]
    text = islam_prompt.format(Category, Question, Answer) + EOS_TOKEN
    return text

fatawas_train = fatawas_train.map(lambda example: {'text': formatting_prompts_func(example)})
fatawas_dev = fatawas_dev.map(lambda example: {'text': formatting_prompts_func(example)})


# Setup trainer for continued pretraining
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=fatawas_train,
    eval_dataset=fatawas_dev,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=3,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        bf16 = is_bfloat16_supported(),
        num_train_epochs = 3,
        eval_strategy =  "steps",
        warmup_steps=5,
        learning_rate=5e-5,
        embedding_learning_rate=1e-5,
        logging_steps=20,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
print("Starting training... Text compilation")
trainer.train()
print("Finished training... Text compilation")


train_batch1 = load_dataset("csv", data_files={"train": "Hirtage_Pipeline_Finetune/data/Task1_MCQs_Train_batch1.csv"}, split="train")

train_batch2 = load_dataset("csv", data_files={"train": "Hirtage_Pipeline_Finetune/data/Task1_MCQs_Train_batch2.csv"}, split="train")

train_dataset = concatenate_datasets([train_batch1, train_batch2])

test_dataset = load_dataset("csv", data_files={"dev": "Hirtage_Pipeline_Finetune/data/Task1_MCQs_Dev.csv"}, split="dev")

print("Train dataset:", train_dataset)
print("Test dataset:", test_dataset)

def format_chat_prompt(example):
    prompt = f''' "ما هي الإجابة الصحيحة على السؤال التالي؟ .\n\n"
        "السؤال: {example['question']}\n"
        الاختيارات :
        "A. {example['option1']}\n"
        "B. {example['option2']}\n"
        "C. {example['option3']}\n"
        "D. {example['option4']}\n"
        "E. {example['option5']}\n"
        "F. {example['option6']}\n"
    جاوب برمز الاجابه الصحيحه فقط
    '''.replace('\n' , '')
    messages =[]
    messages.append({'role' : 'system' , 'content': "أنت مساعد ذكي تتحدث باللغة العربية الفصحى. كن مهنيًا، لبقا وودودا في ردودك، وتحدث دائما بالفصحى. أجب بوضوح وتفصيل، وتجنب الردود المختصرة. يمكنك التحدث بالإنجليزية إذا بدأ المستخدم بها أو طلب ذلك."})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": example['label']})

    # print(messages)
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    return chat_text
train_dataset = train_dataset.map(lambda example: {'text': format_chat_prompt(example)})
test_dataset = test_dataset.map(lambda example: {'text': format_chat_prompt(example)})



trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = UnslothTrainingArguments(
        per_device_train_batch_size =3,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 3,
        warmup_steps = 5,
        bf16 = is_bfloat16_supported(),
        eval_strategy = 'steps',
        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,
        logging_steps = 20,
        optim = "adamw_torch",
        weight_decay = 0.00,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_finetune_after_pretraining",
        load_best_model_at_end=True,
        report_to = "none", # Use this for WandB etc
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)
print("Starting training... Fine-tuning after pretraining")
trainer_stats = trainer.train()
print("Finished training... Fine-tuning after pretraining")
print("Training stats:")
#trainer.evaluate()
model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
