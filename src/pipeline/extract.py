"""VLM extraction via API for structured world state from video frames.

Sends individual frames to a VLM model and parses the structured JSON
response into FrameExtraction objects.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from src.pipeline.ingest import extract_frames
from src.pipeline.schemas import FrameExtraction

if TYPE_CHECKING:
    from src.models.base import BaseModel


_EXTRACTION_PROMPT = """Analyze this image frame and extract structured information.
Return a JSON object with these fields:
- "entities": list of objects with "name" and "position" keys
- "actions": list of objects with "description" and "agent" keys
- "predictions": list of strings describing what might happen next
- "uncertainties": list of strings describing what is unclear

Return ONLY the JSON object, no additional text."""


async def extract_from_frame(
    model: BaseModel,
    frame_bytes: bytes,
    prompt: str = _EXTRACTION_PROMPT,
) -> FrameExtraction:
    """Send a frame to a VLM and parse the structured JSON response.

    Writes the frame to a temporary file, queries the model, and parses
    the response into a FrameExtraction object.

    Args:
        model: A model wrapper that supports image input.
        frame_bytes: PNG-encoded bytes of the frame.
        prompt: The extraction prompt to send with the frame.

    Returns:
        A FrameExtraction with parsed entities, actions, predictions,
        and uncertainties. Returns empty lists on parse failure.
    """
    import tempfile
    from pathlib import Path

    # Write frame to temp file for the model API
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(frame_bytes)
        tmp_path = tmp.name

    try:
        response_text, confidence = await model.query(
            prompt=prompt,
            image_path=tmp_path,
        )

        parsed = _parse_extraction(response_text)

        return FrameExtraction(
            frame_number=0,
            timestamp_s=0.0,
            entities=parsed.get("entities", []),
            actions=parsed.get("actions", []),
            predictions=parsed.get("predictions", []),
            uncertainties=parsed.get("uncertainties", []),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def extract_from_video(
    model: BaseModel,
    video_path: str,
    fps: float = 1.0,
) -> list[FrameExtraction]:
    """Extract structured information from a video by sampling frames.

    Orchestrates frame extraction via decord and VLM queries for each frame.

    Args:
        model: A model wrapper that supports image input.
        video_path: Path to the video file.
        fps: Frames per second to sample (default: 1.0).

    Returns:
        A list of FrameExtraction objects, one per sampled frame.
        Returns an empty list if the video cannot be read.
    """
    frames = extract_frames(video_path, fps=fps)
    if not frames:
        return []

    extractions: list[FrameExtraction] = []

    for frame_number, timestamp_s, frame_bytes in frames:
        extraction = await extract_from_frame(model, frame_bytes)
        # Override frame metadata from the actual extraction position
        extraction.frame_number = frame_number
        extraction.timestamp_s = timestamp_s
        extractions.append(extraction)

    return extractions


def _parse_extraction(text: str) -> dict:
    """Parse a VLM response into a structured extraction dict.

    Attempts to find and parse JSON from the model's response text.
    Falls back to empty structure on parse failure.

    Args:
        text: The raw text response from the VLM.

    Returns:
        A dict with entities, actions, predictions, and uncertainties keys.
    """
    default = {
        "entities": [],
        "actions": [],
        "predictions": [],
        "uncertainties": [],
    }

    # Try to find JSON in the response
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                return {
                    "entities": parsed.get("entities", []),
                    "actions": parsed.get("actions", []),
                    "predictions": parsed.get("predictions", []),
                    "uncertainties": parsed.get("uncertainties", []),
                }
        except json.JSONDecodeError:
            pass

    return default
