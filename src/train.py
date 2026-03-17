"""
train.py
--------
Fine-tuning script for FLAN-T5 on a subtitle quiz-generation dataset.

Training format (instruction tuning):
    Input:  "Generate quiz questions from the following lesson summary:\n{summary}"
    Output: "Question 1: ... Answer: X\nQuestion 2: ... Answer: Y\n..."

Uses HuggingFace Seq2SeqTrainer with automatic train/validation split,
gradient accumulation, mixed-precision (if GPU available), and checkpointing.
"""

import json
import logging
import os
import random
from typing import List, Dict, Any

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Reproducibility seed
# ------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------
TRAIN_INPUT_TEMPLATE = (
    "Generate quiz questions from the following lesson summary:\n{summary}"
)


def prepare_training_samples(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert raw dataset records into instruction-tuning input/output pairs.

    Each record must have:
        - 'chunks': list of raw subtitle strings (used as a naive summary)
        - 'reference_questions': the target quiz text

    If 'reference_questions' is empty, the record is skipped.

    Args:
        data (List[Dict]): Loaded dataset records.

    Returns:
        List of dicts with 'input' and 'output' string keys.
    """
    samples = []
    for entry in data:
        # Use combined chunks as a proxy summary if pre-summarized data unavailable
        summary = " ".join(entry.get("chunks", []))
        target = entry.get("reference_questions", "").strip()

        if not summary or not target:
            logger.warning(
                f"Skipping video_id='{entry.get('video_id')}' — "
                "missing summary or reference questions."
            )
            continue

        samples.append({
            "input": TRAIN_INPUT_TEMPLATE.format(summary=summary),
            "output": target,
        })

    logger.info(f"Prepared {len(samples)} training samples.")
    return samples


def tokenize_dataset(
    samples: List[Dict[str, str]],
    tokenizer: T5Tokenizer,
    max_input_length: int = 512,
    max_target_length: int = 256,
) -> Dataset:
    """
    Tokenize input/output pairs for seq2seq training.

    Labels use -100 for padding tokens so that they are ignored by the loss.

    Args:
        samples: List of {'input': str, 'output': str} dicts.
        tokenizer: T5 tokenizer.
        max_input_length: Max tokens for encoder input.
        max_target_length: Max tokens for decoder target.

    Returns:
        HuggingFace Dataset with 'input_ids', 'attention_mask', 'labels'.
    """
    inputs = [s["input"] for s in samples]
    targets = [s["output"] for s in samples]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding=False,  # DataCollator handles dynamic padding per batch
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )

    # Replace pad token id in labels with -100 to ignore during loss computation
    label_ids = []
    for ids in labels["input_ids"]:
        label_ids.append([
            token_id if token_id != tokenizer.pad_token_id else -100
            for token_id in ids
        ])

    model_inputs["labels"] = label_ids
    return Dataset.from_dict(model_inputs)


def get_training_args(
    output_dir: str,
    num_epochs: int = 5,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 3e-4,
    warmup_steps: int = 50,
    save_steps: int = 100,
    eval_steps: int = 100,
    logging_steps: int = 20,
) -> Seq2SeqTrainingArguments:
    """
    Build Seq2SeqTrainingArguments with sensible defaults.

    Args:
        output_dir (str): Directory for model checkpoints.
        num_epochs (int): Number of training epochs.
        train_batch_size (int): Per-device training batch size.
        eval_batch_size (int): Per-device evaluation batch size.
        gradient_accumulation_steps (int): Accumulate gradients over N steps
            before an optimizer update (effective batch = batch*accum).
        learning_rate (float): Initial optimizer learning rate.
        warmup_steps (int): LR scheduler warm-up steps.
        save_steps (int): Save checkpoint every N steps.
        eval_steps (int): Run evaluation every N steps.
        logging_steps (int): Log metrics every N steps.

    Returns:
        Seq2SeqTrainingArguments instance.
    """
    # Use fp16 if a CUDA GPU is available (speeds up training, saves memory)
    use_fp16 = torch.cuda.is_available()

    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        fp16=use_fp16,
        predict_with_generate=True,   # Required for seq2seq evaluation metrics
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,            # Keep only the 3 most recent checkpoints
        seed=SEED,
        report_to="none",              # Disable W&B / MLflow by default
    )


def run_training(
    data_path: str,
    model_name: str = "google/flan-t5-base",
    output_dir: str = "models/flan-t5-quiz",
    val_split: float = 0.15,
    num_epochs: int = 5,
    train_batch_size: int = 4,
    max_input_length: int = 512,
    max_target_length: int = 256,
) -> None:
    """
    Full end-to-end training pipeline:
        1. Set seed for reproducibility
        2. Load dataset
        3. Prepare and tokenize samples
        4. Split into train / validation
        5. Initialize model, tokenizer, data collator
        6. Configure and run Seq2SeqTrainer
        7. Save final model and tokenizer

    Args:
        data_path (str): Path to the JSON training dataset.
        model_name (str): Base FLAN-T5 model to fine-tune.
        output_dir (str): Directory to save checkpoints and final model.
        val_split (float): Fraction of data reserved for validation.
        num_epochs (int): Training epochs.
        train_batch_size (int): Per-device batch size during training.
        max_input_length (int): Max input token length.
        max_target_length (int): Max label token length.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # ------------------------------------------------------------------ #
    # 1 & 2. Load tokenizer, model & raw data
    # ------------------------------------------------------------------ #
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)

    # ------------------------------------------------------------------ #
    # 3. Prepare data & Tokenize & Split
    # ------------------------------------------------------------------ #
    if os.path.exists(data_path) and os.path.isdir(data_path):
        logger.info(f"Loading split dataset from directory: {data_path}")
        train_file = os.path.join(data_path, "train.json")
        val_file = os.path.join(data_path, "validation.json")
        
        with open(train_file, "r", encoding="utf-8") as f:
            train_raw = json.load(f)
        with open(val_file, "r", encoding="utf-8") as f:
            val_raw = json.load(f)
            
        train_samples = prepare_training_samples(train_raw)
        val_samples = prepare_training_samples(val_raw)
        
        if not train_samples or not val_samples:
            raise ValueError("No valid training/validation samples found.")
            
        train_dataset = tokenize_dataset(train_samples, tokenizer, max_input_length, max_target_length)
        eval_dataset = tokenize_dataset(val_samples, tokenizer, max_input_length, max_target_length)
    else:
        logger.info(f"Loading dataset from file: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = prepare_training_samples(raw_data)
        if not samples:
            raise ValueError("No valid training samples found. Check dataset format.")
            
        full_dataset = tokenize_dataset(
            samples, tokenizer, max_input_length, max_target_length
        )

        split = full_dataset.train_test_split(test_size=val_split, seed=SEED)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    logger.info(
        f"Dataset split — train: {len(train_dataset)}, eval: {len(eval_dataset)}"
    )

    # ------------------------------------------------------------------ #
    # 4. Data collator (dynamic padding per batch)
    # ------------------------------------------------------------------ #
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    # ------------------------------------------------------------------ #
    # 5. Training arguments & Trainer
    # ------------------------------------------------------------------ #
    os.makedirs(output_dir, exist_ok=True)
    training_args = get_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        train_batch_size=train_batch_size,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ------------------------------------------------------------------ #
    # 6. Train
    # ------------------------------------------------------------------ #
    logger.info("Starting training ...")
    trainer.train()

    # ------------------------------------------------------------------ #
    # 7. Save final model
    # ------------------------------------------------------------------ #
    final_model_path = os.path.join(output_dir, "final")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
