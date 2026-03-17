"""
main.py
--------
CLI entry point for the FLAN-T5 Quiz Generation project.

Usage:
    # Run inference on the sample dataset (default):
    python main.py

    # Run fine-tuning on the sample dataset:
    python main.py --train

    # Specify a custom dataset:
    python main.py --data path/to/your_dataset.json

    # Specify a fine-tuned model for inference:
    python main.py --model models/flan-t5-quiz/final
"""

import argparse
import json
import logging
import sys

from src.preprocessing import load_dataset_json, extract_chunks
from src.inference import generate_quiz_from_subtitles, evaluate_output, quiz_to_json_string

# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
DEFAULT_DATA_PATH = "data/splits"
DEFAULT_MODEL = "google/flan-t5-base"
DEFAULT_OUTPUT_DIR = "models/flan-t5-quiz"


def run_inference(data_path: str, model_name: str) -> None:
    """
    Load the first video entry from the dataset, run the full
    quiz-generation pipeline, and print the result as JSON.

    Also evaluates against reference_questions if present.

    Args:
        data_path (str): Path to the JSON dataset.
        model_name (str): HuggingFace model identifier or local path.
    """
    logger.info(f"Loading dataset: {data_path}")
    dataset = load_dataset_json(data_path)

    if not dataset:
        logger.error("Dataset is empty. Exiting.")
        sys.exit(1)

    # ---- Process each video entry ----
    for entry in dataset:
        video_id = entry["video_id"]
        chunks = extract_chunks(entry)
        reference = entry.get("reference_questions", "")

        print(f"\n{'=' * 60}")
        print(f"  Video ID : {video_id}")
        print(f"  Chunks   : {len(chunks)}")
        print(f"{'=' * 60}\n")

        # Run the full pipeline
        result = generate_quiz_from_subtitles(
            subtitle_chunks=chunks,
            model_name=model_name,
        )

        # Pretty-print the quiz JSON
        print("📚  LESSON SUMMARY")
        print("-" * 40)
        print(result["summary"])
        print()

        print("❓  GENERATED QUIZ")
        print("-" * 40)
        for i, q in enumerate(result["questions"], start=1):
            print(f"Question {i}: {q['question']}")
            option_labels = ["A", "B", "C", "D"]
            for label, opt in zip(option_labels, q.get("options", [])):
                print(f"  {label}. {opt}")
            print(f"  ✅ Answer: {q.get('answer', 'N/A')}")
            print()

        # Optional: ROUGE evaluation against reference
        if reference:
            # Flatten generated questions back to a single string for comparison
            generated_flat = " ".join(
                f"{q['question']} " + " ".join(q.get("options", []))
                for q in result["questions"]
            )
            scores = evaluate_output(reference=reference, hypothesis=generated_flat)
            print("📊  ROUGE Evaluation Scores")
            print("-" * 40)
            for metric, score in scores.items():
                print(f"  {metric}: {score:.4f}")
            print()

        # Also output raw JSON
        print("🗂️  Raw JSON Output")
        print("-" * 40)
        print(quiz_to_json_string(result))
        print()


def run_training(data_path: str, model_name: str, output_dir: str) -> None:
    """
    Fine-tune FLAN-T5 on the provided dataset.

    Args:
        data_path (str): Path to the JSON training dataset.
        model_name (str): Base model to fine-tune.
        output_dir (str): Directory to save checkpoints and final model.
    """
    # Import here to avoid loading heavy training deps during inference
    from src.train import run_training as _run_training  # noqa: F401

    logger.info(f"Starting fine-tuning: model={model_name}, data={data_path}")
    _run_training(
        data_path=data_path,
        model_name=model_name,
        output_dir=output_dir,
    )
    logger.info(f"Training complete. Model saved to: {output_dir}")


# ------------------------------------------------------------------
# CLI argument parser
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FLAN-T5 Quiz Generator from Video Subtitles"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to JSON subtitle dataset (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Checkpoint directory for training (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run fine-tuning instead of inference.",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.train:
        run_training(
            data_path=args.data,
            model_name=args.model,
            output_dir=args.output_dir,
        )
    else:
        run_inference(
            data_path=args.data,
            model_name=args.model,
        )
