import torch
from datasets import load_dataset
import re
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import pandas as pd
import re
import zipfile

# 1. Specify your checkpoint directory
#    This is the same `output_dir` you passed to the trainer,
#    plus the checkpoint name (e.g. “checkpoint-125”).
checkpoint_dir = "/unsloth/outputs_finetune_after_pretraining/checkpoint-1500"  # Change this to your actual checkpoint directory


model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_dir,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference    
device = "cuda" if torch.cuda.is_available() else "cpu"


letter_choices = ["A", "B", "C", "D","E","F"]
max_new_tokens = 5  # enough for a full sentence answer



letter_choices = ["A", "B", "C", "D", "E", "F"]
max_new_tokens = 5
team_name = "Gumball"


dataset = load_dataset("csv", data_files={"dev": "/unsloth/data/Task1_MCQ_Test.csv"}, split="dev")
subset = dataset

submission_data = []

def format_chat_prompt(example):
    prompt = f'''
    ما هي الإجابة الصحيحة على السؤال التالي؟
    السؤال: {example['question']}
    الاختيارات:
    A. {example['option1']}
    B. {example['option2']}
    C. {example['option3']}
    D. {example['option4']}
    E. {example['option5']}
    F. {example['option6']}
    جاوب برمز الإجابة الصحيحة فقط.
    '''.replace('\n', ' ')

    messages = [
        {'role': 'system', 'content': "أنت مساعد ذكي تتحدث باللغة العربية الفصحى. كن مهنيًا، لبقا وودودا في ردودك."},
        {'role': 'user', 'content': prompt}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return chat_text

for idx, example in enumerate(subset):
    chat_prompt = format_chat_prompt(example)
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    predicted_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(idx)
    print(predicted_text)

    match = re.search(r"\b([A-F])\b", predicted_text.upper())
    prediction = match.group(1) if match else "F"
    # حفظ النتيجة
    submission_data.append({
        "id_question": example["id_question"],
        "prediction": prediction
    })

# إنشاء ملف CSV
df_submission = pd.DataFrame(submission_data)
csv_filename = f"subtask1_{team_name}_predictions.csv"
df_submission.to_csv(csv_filename, index=False, encoding="utf-8")

# ضغط الملف
zip_filename = f"subtask1_{team_name}_predictions.zip"
with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(csv_filename)

print(f"\n✅ Submission file created: {zip_filename}")
