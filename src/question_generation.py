"""
question_generation.py
-----------------------
Uses FLAN-T5 to generate multiple-choice quiz questions from a combined
lesson summary, then parses the raw text output into structured dicts.
"""

import re
import logging
from typing import List, Dict, Any

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Prompt template for quiz generation
# ------------------------------------------------------------------
QUESTION_GEN_PROMPT = (
    "Generate 3 multiple choice questions with 4 options each (A, B, C, D) "
    "and indicate the correct answer based on the following lesson summary.\n\n"
    "Format each question exactly as:\n"
    "Question N:\n"
    "A. ...\n"
    "B. ...\n"
    "C. ...\n"
    "D. ...\n"
    "Answer: X\n\n"
    "Lesson summary: {summary}"
)


def generate_questions(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    summary: str,
    device: torch.device,
    num_beams: int = 5,
    max_input_length: int = 512,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate multiple-choice quiz questions from a summary using FLAN-T5.

    Args:
        model: Loaded T5 model.
        tokenizer: Corresponding T5 tokenizer.
        summary (str): Combined lesson summary to base questions on.
        device (torch.device): Device for inference.
        num_beams (int): Beam search width.
        max_input_length (int): Token limit for the input prompt.
        max_new_tokens (int): Maximum tokens in the generated output.

    Returns:
        Raw generated text string containing all questions.
    """
    prompt = QUESTION_GEN_PROMPT.format(summary=summary)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
        padding=False,
    ).to(device)

    logger.info("Generating quiz questions ...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=3,   # Prevent repetitive phrasing
            early_stopping=True,
        )

    raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    logger.debug(f"Raw generated output:\n{raw_text}")
    return raw_text


def parse_questions(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse raw FLAN-T5 output into a list of structured question dicts.

    Each dict has the shape:
        {
            "question": str,
            "options": [str, str, str, str],   # [A, B, C, D]
            "answer": str                        # e.g. "B"
        }

    The parser is lenient — it handles minor formatting deviations and
    falls back to empty fields rather than crashing.

    Args:
        raw_text (str): Raw text output from the model.

    Returns:
        List of question dicts. May be shorter than 3 if parsing fails.
    """
    questions = []

    # Split on "Question N:" boundaries (handles "Question 1:" ... "Question 3:")
    blocks = re.split(r"Question\s+\d+\s*:", raw_text, flags=re.IGNORECASE)

    # First element before the first "Question" header is discarded
    for block in blocks[1:]:
        block = block.strip()
        if not block:
            continue

        question_dict: Dict[str, Any] = {
            "question": "",
            "options": [],
            "answer": "",
        }

        lines = [line.strip() for line in block.splitlines() if line.strip()]

        # The first line is the question text
        if lines:
            question_dict["question"] = lines[0]

        # Parse options: lines starting with A. / B. / C. / D.
        option_pattern = re.compile(r"^([A-D])\.\s*(.*)", re.IGNORECASE)
        for line in lines[1:]:
            match = option_pattern.match(line)
            if match:
                question_dict["options"].append(match.group(2).strip())

        # Parse answer: "Answer: X"
        answer_pattern = re.compile(r"Answer\s*:\s*([A-D])", re.IGNORECASE)
        answer_match = answer_pattern.search(block)
        if answer_match:
            question_dict["answer"] = answer_match.group(1).upper()

        # Only include entries that have at least a question and some options
        if question_dict["question"] and question_dict["options"]:
            questions.append(question_dict)

    if not questions:
        logger.warning(
            "Could not parse any questions from the model output. "
            "Returning raw text as a single entry."
        )
        questions.append({
            "question": raw_text,
            "options": [],
            "answer": "",
        })

    return questions
