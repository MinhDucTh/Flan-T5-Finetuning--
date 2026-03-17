# FLAN-T5 Quiz Generator from Video Subtitles

Automatically generate **3 multiple-choice quiz questions** from educational video subtitle chunks using **FLAN-T5** (`google/flan-t5-base`).

---

## Pipeline

```
Subtitle Chunks
     │
     ▼
Chunk 1 → Summary 1
Chunk 2 → Summary 2     (FLAN-T5 Summarization)
Chunk 3 → Summary 3
     │
     ▼
Combined Summary
     │
     ▼
FLAN-T5 Question Generation
     │
     ▼
3 Multiple-Choice Questions (JSON)
```

---

## Project Structure

```
Flan-T5/
│
├── data/
│   └── sample_dataset.json       # Example subtitle dataset
│
├── models/                       # Fine-tuned model checkpoints (created on training)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Dataset loading & validation
│   ├── summarization.py          # Chunk summarization via FLAN-T5
│   ├── question_generation.py    # Quiz generation & output parsing
│   ├── train.py                  # Fine-tuning with Seq2SeqTrainer
│   └── inference.py              # End-to-end pipeline + BLEU/ROUGE evaluation
│
├── main.py                       # CLI entry point
└── requirements.txt
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional — for BLEU scoring) Download NLTK data

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
```

---

## Usage

### Run Inference (default)

Runs the full pipeline on `data/sample_dataset.json` using the base `google/flan-t5-base` model:

```bash
python main.py
```

### Run Inference with Custom Dataset or Model

```bash
python main.py --data path/to/your_dataset.json --model google/flan-t5-base
```

Use a fine-tuned local model:

```bash
python main.py --model models/flan-t5-quiz/final
```

### Fine-Tune FLAN-T5

```bash
python main.py --train --data data/sample_dataset.json --output-dir models/flan-t5-quiz
```

---

## Dataset Format (JSON)

```json
[
  {
    "video_id": "video_01",
    "chunks": [
      "chunk text 1",
      "chunk text 2",
      "chunk text 3"
    ],
    "reference_questions": "Question 1: ...\nAnswer: B\n..."
  }
]
```

`reference_questions` is **optional** — used only for ROUGE evaluation during inference.

---

## Output Format

```json
{
  "summary": "Machine learning is a subset of AI. ...",
  "questions": [
    {
      "question": "What is machine learning?",
      "options": [
        "A programming language",
        "A subset of AI that learns from data",
        "A type of database",
        "A hardware device"
      ],
      "answer": "B"
    }
  ]
}
```

---

## Using the API Directly

```python
from src.inference import generate_quiz_from_subtitles

chunks = [
    "In this lecture, we explore neural networks ...",
    "Backpropagation is used to train neural networks ...",
    "CNNs are specialized for image data ...",
]

result = generate_quiz_from_subtitles(subtitle_chunks=chunks)
print(result["summary"])
for q in result["questions"]:
    print(q["question"])
    print(q["options"])
    print("Answer:", q["answer"])
```

---

## Evaluation

ROUGE scores are automatically computed during inference if `reference_questions` is present in your dataset.

Manual BLEU scoring:

```python
from src.inference import compute_bleu

score = compute_bleu(reference="...", hypothesis="...")
print(f"BLEU: {score:.4f}")
```

---

## Training Arguments (key defaults)

| Argument | Default | Description |
|---|---|---|
| `num_epochs` | 5 | Training epochs |
| `train_batch_size` | 4 | Per-device batch size |
| `gradient_accumulation_steps` | 4 | Effective batch = 16 |
| `learning_rate` | 3e-4 | AdamW learning rate |
| `fp16` | Auto (GPU only) | Mixed precision training |
| `val_split` | 0.15 | 15% held out for validation |
| `save_total_limit` | 3 | Max checkpoints retained |

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- Transformers ≥ 4.38
- CUDA GPU strongly recommended for training (CPU inference works but is slow)
