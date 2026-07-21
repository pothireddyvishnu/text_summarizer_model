# 🧠 Text Summarizer Model

Fine-tune a **T5-small** transformer model for text summarization using the **SAMSum** dialogue dataset. This repository
contains the training notebook, dataset, and the resulting saved model.

---

## 🌟 Overview

This project fine-tunes Google's [T5-small](https://huggingface.co/google-t5/t5-small) model on the **SAMSum Corpus** — a dataset
of ~16,000 messenger-like conversations with human-written summaries. The trained model can summarize dialogues,
conversations, and general text into concise summaries.

---

## 🏗️ Project Structure

```
Text Summarizer Model/
├── text_summarizer.ipynb    # Jupyter notebook (full training pipeline)
├── samsum-train.csv         # Training dataset (14,732 samples)
├── samsum-validation.csv    # Validation dataset (818 samples)
├── samsum-test.csv          # Test dataset
├── requirements.txt         # Python dependencies
├── .gitignore               # Git exclusion rules
├── saved_summary_model/     # Exported fine-tuned model & tokenizer
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── results/                 # Training checkpoints
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── checkpoint-1500/
│   ├── checkpoint-2000/
│   ├── checkpoint-2500/
│   └── checkpoint-3000/
└── README.md
```

---

## 📋 Prerequisites

- **Python** 3.10+
- **pip** (Python package manager)
- **Jupyter Notebook** or **JupyterLab**
- **Git** (for cloning)
- **GPU (Recommended)** — Apple MPS / NVIDIA CUDA for faster training. CPU works but will be significantly slower.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/pothireddyvishnu/text_summarizer_model.git
cd text_summarizer_model
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip3 install -r requirements.txt
```

| Package        | Purpose                                          |
|----------------|--------------------------------------------------|
| `pandas`       | Loading and preprocessing the CSV datasets       |
| `transformers` | HuggingFace T5 model, tokenizer, and Trainer API |
| `torch`        | PyTorch deep learning backend                    |
| `jupyter`      | Running the training notebook                    |

### 4. Launch the Notebook

```bash
jupyter notebook text_summarizer.ipynb
```

---

## 🏋️ Training Pipeline

The notebook (`text_summarizer.ipynb`) walks through the full pipeline:

### Step 1 — Load Data

```python
train_data = pd.read_csv("samsum-train.csv")  # 14,732 samples
val_data = pd.read_csv("samsum-validation.csv")  # 818 samples
```

Each row contains: `id`, `dialogue`, and `summary`.

### Step 2 — Data Preprocessing

- Drop rows with missing `dialogue` or `summary` values.
- Clean text by removing line breaks (`\r\n`), extra whitespace, and HTML tags.
- Lowercase all text.

```python
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text
```

### Step 3 — Tokenization

- Load the `T5Tokenizer` from the pretrained `t5-small` model.
- Tokenize dialogues (max 512 tokens) and summaries (max 150 tokens) with padding and truncation.

### Step 4 — Model Fine-Tuning

- Load the pretrained `T5ForConditionalGeneration` (`t5-small`).
- Automatically detect and use the best available device (MPS / CUDA / CPU).

**Training Configuration:**

| Hyperparameter      | Value     |
|---------------------|-----------|
| Epochs              | 6         |
| Train Batch Size    | 8         |
| Eval Batch Size     | 8         |
| Weight Decay        | 0.01      |
| Warmup Steps        | 500       |
| Evaluation Strategy | Per Epoch |
| Save Strategy       | Per Epoch |

```python
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=6,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=500
)
```

### Step 5 — Save the Model

After training completes, the fine-tuned model and tokenizer are saved:

```python
model.save_pretrained("./saved_summary_model")
tokenizer.save_pretrained("./saved_summary_model")
```

### Step 6 — Test the Model

The notebook includes a test section where you can pass any text and get a summary:

```python
summary = summarize_dialogue("Your text here...")
print("Summary:", summary)
```

**Example Output:**

> *Input:* A long article about AI adoption, ethical concerns, and regulation...
>
> *Summary:*
`ai adoption has significantly increased over the past few years. experts highlight the importance of responsible ai development, including data privacy, security and long-term societal impact.`

---

## 📦 Exporting the Model to the App

The trained model in `saved_summary_model/` is used by the companion web application:

**[Text Summarizer App](https://github.com/pothireddyvishnu/text_summarizer_app.git)**

To use the model with the app, copy the `saved_summary_model/` directory:

```bash
cp -r ./saved_summary_model /path/to/text_summarizer_app/saved_summary_model
```

---

## 📊 Dataset

This project uses the **SAMSum Corpus** — a collection of messenger-like conversations with human-annotated summaries.

| Split      | Samples                  |
|------------|--------------------------|
| Train      | 14,732                   |
| Validation | 818                      |
| Test       | Available for evaluation |

---

## 🛠️ Tech Stack

- **Model:** T5-small (60M parameters)
- **Framework:** PyTorch + HuggingFace Transformers
- **Training:** HuggingFace `Trainer` API
- **Dataset:** SAMSum Corpus

---

## 🔗 Related Repository

| Repository                                                                    | Description                                                        |
|-------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [Text Summarizer App](https://github.com/pothireddyvishnu/text_summarizer_app.git) | FastAPI web app that serves this model for real-time summarization |

---
