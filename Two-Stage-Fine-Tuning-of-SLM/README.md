# 📖 Islamic Legal Text Modeling – End-to-End Pipeline

Our end-to-end system  is organized as three distinct stages executed in sequence:

1. **Stage 0: Preprocessing (Optional)** – Clean datasets by removing Arabic diacritics (tashkeel) and unnecessary punctuation.
2. **Stage 1: Continued Pretraining** – Domain-adaptive training on the Fatwa corpus (LoRA on Qwen3-4B).
3. **Stage 2: Instruction Fine-tuning** – Training on Islamic inheritance MCQ datasets.
4. **Stage 3: Evaluation** – Running inference on test questions and exporting results.

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone git@github.com:Aml-Hassan-Abd-El-hamid/Islamic-inheritance-using-AI-qias-2025.git -b Two-Stage-Fine-Tuning-of-SLM
pip install -r requirments.txt
```

---

## 📂 Files Overview

### 🔹 Core Pipeline Scripts

* **`text_compilation.py`**

  * Handles both **continued pretraining** (Fatwa corpus) and **instruction fine-tuning** (MCQ datasets).
  * Formats datasets into structured prompts.
  * Saves the final LoRA-adapted model.

* **`evaluation.py`**

  * Loads the trained checkpoint from Stage 2.
  * Generates predictions on MCQ test data.
  * Saves predictions to **CSV** and compresses them into a **ZIP file** (competition-ready submission).

### 🔹 Preprocessing Script

* **`cleanup_tashkeel.py`**

  * Cleans Arabic text by removing **diacritics (tashkeel)** and unnecessary punctuation.
  * Preserves essential characters: `؟` (Arabic question mark), `()`, and `.`.
  * Input: `data/Task1_MCQ_Test.csv`
  * Output: `data/Task1_MCQ_Test_cleaned_no_tashkeel.csv`

👉 Use this step if you want to work with a **normalized, diacritic-free dataset** before training or evaluation.

### 🔹 Environment Setup

* **`requirments.txt`**

  * Contains all required dependencies:

    * Model training: `torch`, `transformers`, `unsloth`, `peft`, `trl`
    * Dataset handling: `datasets`, `protobuf`, `sentencepiece`
    * Efficiency: `xformers`, `bitsandbytes`, `hf_transfer`
    * Training acceleration: `accelerate`

---

## 🚀 Usage

### **Stage 0: Preprocessing (Optional)**

Run the text cleaning script:

```bash
python cleanup_tashkeel.py
```

This will save a cleaned CSV without tashkeel to `data/Task1_MCQ_Test_cleaned_no_tashkeel.csv`.

---

### **Stage 1 + Stage 2: Training**

Run the combined training pipeline:

```bash
python text_compilation.py
```

This will:

* Continue pretraining the model on the Fatwa corpus.
* Fine-tune the model on Islamic inheritance MCQs.
* Save outputs to:

  * Checkpoints → `outputs_finetune_after_pretraining/`
  * Final model → `lora_model/`

---

### **Stage 3: Evaluation & Submission**

Evaluate the fine-tuned model on the test set:

```bash
python evaluation.py
```

This will:

* Load the latest checkpoint.
* Predict answers for the test dataset.
* Save results into:

  * `subtask1_<team_name>_predictions.csv`
  * `subtask1_<team_name>_predictions.zip`

---

## 📊 Pipeline Summary

```text
Stage 0 → Preprocessing (cleanup_tashkeel.py) [Optional]
Stage 1 → Continued Pretraining (Fatwa corpus)
Stage 2 → Instruction Fine-tuning (Inheritance MCQs)
Stage 3 → Evaluation & Submission
```


