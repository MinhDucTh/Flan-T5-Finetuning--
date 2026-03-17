"""
summarization.py
----------------
Uses FLAN-T5 to summarize individual subtitle chunks, then combines
them into a single unified lesson summary for downstream quiz generation.
"""

import logging
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Prompt template for summarization
# ------------------------------------------------------------------
SUMMARIZATION_PROMPT = "Summarize the following lecture content briefly: {chunk}"


def get_device() -> torch.device:
    """
    Return the best available device (CUDA GPU if available, else CPU).

    Returns:
        torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def load_summarizer(
    model_name: str = "google/flan-t5-base",
    device: torch.device = None,
) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """
    Load the FLAN-T5 tokenizer and model, move the model to the target device.

    Args:
        model_name (str): HuggingFace model identifier.
        device (torch.device): Target device. If None, auto-detected.

    Returns:
        Tuple of (model, tokenizer).
    """
    if device is None:
        device = get_device()

    logger.info(f"Loading tokenizer from '{model_name}' ...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    logger.info(f"Loading model from '{model_name}' ...")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    model.eval()  # Disable dropout for inference

    logger.info("Model loaded and moved to device.")
    return model, tokenizer


def summarize_chunk(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    chunk: str,
    device: torch.device,
    max_input_length: int = 512,
    max_new_tokens: int = 128,
    num_beams: int = 4,
) -> str:
    """
    Summarize a single text chunk using FLAN-T5.

    Args:
        model: Loaded T5 model.
        tokenizer: Corresponding T5 tokenizer.
        chunk (str): Raw lecture text to summarize.
        device (torch.device): Device for tensor operations.
        max_input_length (int): Maximum token length for the input prompt.
        max_new_tokens (int): Maximum tokens to generate in the summary.
        num_beams (int): Beam search width — higher = more coherent but slower.

    Returns:
        Summary string.
    """
    prompt = SUMMARIZATION_PROMPT.format(chunk=chunk)

    # Tokenize the prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
        padding=False,
    ).to(device)

    # Generate without computing gradients (saves memory)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

    # Decode and clean up the generated text
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary.strip()


def summarize_all_chunks(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    chunks: List[str],
    device: torch.device,
    max_input_length: int = 512,
    max_new_tokens: int = 128,
    num_beams: int = 4,
) -> List[str]:
    """
    Summarize each chunk in a list, showing a progress bar.

    Args:
        model: Loaded T5 model.
        tokenizer: Corresponding T5 tokenizer.
        chunks (List[str]): List of raw lecture chunks.
        device (torch.device): Device for tensor operations.
        max_input_length (int): Token limit for each input prompt.
        max_new_tokens (int): Token limit for each generated summary.
        num_beams (int): Beam search width.

    Returns:
        List of summary strings (one per chunk, in order).
    """
    summaries = []
    logger.info(f"Summarizing {len(chunks)} chunks ...")

    for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
        summary = summarize_chunk(
            model=model,
            tokenizer=tokenizer,
            chunk=chunk,
            device=device,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        logger.debug(f"  Chunk {i + 1} summary: {summary[:80]}...")
        summaries.append(summary)

    return summaries


def combine_summaries(summaries: List[str], separator: str = " ") -> str:
    """
    Concatenate individual chunk summaries into a single combined summary.

    Args:
        summaries (List[str]): Ordered list of chunk summaries.
        separator (str): String used to join summaries (default: single space).

    Returns:
        Combined summary string.
    """
    if not summaries:
        raise ValueError("Cannot combine an empty list of summaries.")

    combined = separator.join(s.strip() for s in summaries if s.strip())
    logger.info(f"Combined summary length: {len(combined)} characters.")
    return combined
