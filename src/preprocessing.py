"""
preprocessing.py
----------------
Handles loading and validating subtitle datasets from JSON or CSV files.
Each record must have a 'video_id' and a 'chunks' list of text segments.
"""

import json
import csv
import logging
from typing import List, Dict, Any

# Configure module-level logger
logger = logging.getLogger(__name__)


def load_dataset_json(path: str) -> List[Dict[str, Any]]:
    """
    Load subtitle dataset from a JSON file.

    Expected format:
    [
        {
            "video_id": "video_01",
            "chunks": ["chunk text 1", "chunk text 2", ...],
            "reference_questions": "..."   # optional, used for evaluation
        },
        ...
    ]

    Args:
        path (str): Path to the JSON dataset file.

    Returns:
        List of dicts, one per video entry.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON structure is invalid.
    """
    logger.info(f"Loading JSON dataset from: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {path}: {e}")

    if not isinstance(data, list):
        raise ValueError("JSON dataset must be a list of video objects.")

    # Validate each entry
    validated = []
    for i, entry in enumerate(data):
        video_id = entry.get("video_id", f"unknown_{i}")
        chunks = entry.get("chunks", [])
        chunks = validate_chunks(chunks, video_id)
        validated.append({
            "video_id": video_id,
            "chunks": chunks,
            "reference_questions": entry.get("reference_questions", "")
        })

    logger.info(f"Loaded {len(validated)} video entries.")
    return validated


def load_dataset_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load subtitle dataset from a CSV file.

    Expected CSV columns: video_id, chunk_index, chunk_text
    Rows with the same video_id are grouped into a single entry.

    Args:
        path (str): Path to the CSV dataset file.

    Returns:
        List of dicts with 'video_id' and 'chunks' keys.
    """
    logger.info(f"Loading CSV dataset from: {path}")
    video_map: Dict[str, List[str]] = {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row.get("video_id", "unknown")
                chunk = row.get("chunk_text", "").strip()
                if vid not in video_map:
                    video_map[vid] = []
                if chunk:
                    video_map[vid].append(chunk)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {path}")

    result = []
    for vid, chunks in video_map.items():
        chunks = validate_chunks(chunks, vid)
        result.append({"video_id": vid, "chunks": chunks, "reference_questions": ""})

    logger.info(f"Loaded {len(result)} video entries from CSV.")
    return result


def validate_chunks(chunks: List[str], video_id: str = "") -> List[str]:
    """
    Validate and clean a list of text chunks.

    - Removes empty or whitespace-only strings.
    - Warns if chunks are very short (< 20 characters) — may indicate bad data.

    Args:
        chunks (List[str]): Raw chunk list.
        video_id (str): Identifier used for log messages.

    Returns:
        Cleaned list of non-empty chunk strings.
    """
    if not isinstance(chunks, list):
        logger.warning(f"[{video_id}] 'chunks' is not a list. Returning empty.")
        return []

    cleaned = []
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, str):
            logger.warning(f"[{video_id}] Chunk {i} is not a string, skipping.")
            continue
        chunk = chunk.strip()
        if not chunk:
            logger.warning(f"[{video_id}] Chunk {i} is empty, skipping.")
            continue
        if len(chunk) < 20:
            logger.warning(
                f"[{video_id}] Chunk {i} is very short ({len(chunk)} chars): '{chunk}'"
            )
        cleaned.append(chunk)

    if not cleaned:
        logger.error(f"[{video_id}] No valid chunks found after validation.")

    return cleaned


def extract_chunks(entry: Dict[str, Any]) -> List[str]:
    """
    Convenience function: extract the chunks list from a dataset entry.

    Args:
        entry (Dict): A single dataset record.

    Returns:
        List of chunk strings.
    """
    return entry.get("chunks", [])
