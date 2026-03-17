"""
inference.py
-------------
Public-facing end-to-end pipeline.

Entry point: generate_quiz_from_subtitles(subtitle_chunks)

Pipeline:
    1. Load FLAN-T5 model and tokenizer
    2. Summarize each subtitle chunk individually
    3. Combine chunk summaries into one lesson summary
    4. Generate 3 multiple-choice quiz questions from the combined summary
    5. Parse raw output into structured JSON
    6. (Optional) Evaluate against reference answers with BLEU/ROUGE

Returns a JSON-ready dict:
    {
        "summary": "...",
        "questions": [
            {
                "question": "...",
                "options": ["...", "...", "...", "..."],
                "answer": "A"
            },
            ...
        ]
    }
"""

import json
import logging
from typing import List, Dict, Any, Optional

import torch

from src.summarization import (
    get_device,
    load_summarizer,
    summarize_all_chunks,
    combine_summaries,
)
from src.question_generation import generate_questions, parse_questions

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def generate_quiz_from_subtitles(
    subtitle_chunks: List[str],
    model_name: str = "google/flan-t5-base",
    device: Optional[torch.device] = None,
    summarizer_max_new_tokens: int = 128,
    question_max_new_tokens: int = 512,
    num_beams: int = 4,
) -> Dict[str, Any]:
    """
    Full pipeline: subtitle chunks → structured quiz JSON.

    Args:
        subtitle_chunks (List[str]): List of raw subtitle text segments.
        model_name (str): HuggingFace model to use (default: flan-t5-base).
        device (torch.device | None): Target device; auto-detected if None.
        summarizer_max_new_tokens (int): Max tokens per chunk summary.
        question_max_new_tokens (int): Max tokens for quiz generation output.
        num_beams (int): Beam search width used in both steps.

    Returns:
        Dict with 'summary' (str) and 'questions' (List[Dict]).

    Raises:
        ValueError: If subtitle_chunks is empty.
    """
    if not subtitle_chunks:
        raise ValueError("subtitle_chunks must not be empty.")

    # ------------------------------------------------------------------ #
    # Step 1: Resolve device and load model
    # ------------------------------------------------------------------ #
    if device is None:
        device = get_device()

    logger.info("Loading model and tokenizer for inference ...")
    model, tokenizer = load_summarizer(model_name=model_name, device=device)

    # ------------------------------------------------------------------ #
    # Step 2: Summarize each chunk
    # ------------------------------------------------------------------ #
    chunk_summaries = summarize_all_chunks(
        model=model,
        tokenizer=tokenizer,
        chunks=subtitle_chunks,
        device=device,
        max_new_tokens=summarizer_max_new_tokens,
        num_beams=num_beams,
    )

    # ------------------------------------------------------------------ #
    # Step 3: Combine summaries into one lesson summary
    # ------------------------------------------------------------------ #
    combined_summary = combine_summaries(chunk_summaries)
    logger.info(f"Combined summary: {combined_summary[:120]} ...")

    # ------------------------------------------------------------------ #
    # Step 4: Generate quiz questions
    # ------------------------------------------------------------------ #
    raw_output = generate_questions(
        model=model,
        tokenizer=tokenizer,
        summary=combined_summary,
        device=device,
        num_beams=num_beams,
        max_new_tokens=question_max_new_tokens,
    )

    # ------------------------------------------------------------------ #
    # Step 5: Parse output into structured dicts
    # ------------------------------------------------------------------ #
    questions = parse_questions(raw_output)

    result = {
        "summary": combined_summary,
        "questions": questions,
    }

    logger.info(f"Generated {len(questions)} question(s).")
    return result


# ------------------------------------------------------------------
# Optional evaluation helpers
# ------------------------------------------------------------------

def evaluate_output(
    reference: str,
    hypothesis: str,
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores between a reference string
    and a hypothesis string. Useful for comparing generated quizzes against
    human-written reference questions.

    Requires: pip install rouge-score

    Args:
        reference (str): Ground-truth text (e.g. reference quiz questions).
        hypothesis (str): Model-generated text to evaluate.

    Returns:
        Dict mapping metric name → F1 score (0.0 – 1.0).
    """
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError:
        raise ImportError(
            "rouge-score is not installed. Run: pip install rouge-score"
        )

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)

    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Compute corpus BLEU score between a reference and hypothesis string.

    Uses NLTK's sentence_bleu with smoothing function 4.

    Requires: pip install nltk

    Args:
        reference (str): Ground-truth text.
        hypothesis (str): Model-generated text.

    Returns:
        BLEU score as a float (0.0 – 1.0).
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
        from nltk.tokenize import word_tokenize  # type: ignore
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except ImportError:
        raise ImportError("nltk is not installed. Run: pip install nltk")

    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    smoother = SmoothingFunction().method4
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)
    return round(score, 4)


def quiz_to_json_string(quiz: Dict[str, Any], indent: int = 2) -> str:
    """Serialize a quiz result dict to a pretty-printed JSON string."""
    return json.dumps(quiz, indent=indent, ensure_ascii=False)
