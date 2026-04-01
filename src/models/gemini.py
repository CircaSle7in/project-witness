"""Gemini Flash API wrapper using the google-genai SDK.

Reads GEMINI_API_KEY from the environment. Supports text, image, and video
inputs. Also serves as the primary LLM judge for evaluation scoring.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from google import genai
from google.genai import types

from src.models.base import BaseModel


class GeminiModel(BaseModel):
    """Wrapper for Google Gemini Flash via the google-genai SDK."""

    def __init__(self, model_id: str = "gemini-2.5-flash") -> None:
        """Initialize the Gemini client.

        Args:
            model_id: The Gemini model identifier to use.

        Raises:
            ValueError: If GEMINI_API_KEY is not set in the environment.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.name = f"gemini-flash-{model_id.split('-')[-1]}"

    async def query(
        self,
        prompt: str,
        image_path: str | None = None,
        video_path: str | None = None,
    ) -> tuple[str, float]:
        """Send a prompt with optional media to Gemini Flash.

        Asks the model to include a confidence score in its structured output.

        Args:
            prompt: The text prompt to send.
            image_path: Optional path to an image file.
            video_path: Optional path to a video file.

        Returns:
            A tuple of (response_text, confidence_score).
        """
        structured_prompt = (
            f"{prompt}\n\n"
            "After your answer, on a new line, provide your confidence as a JSON "
            'object: {{"confidence": <float 0-1>}}'
        )

        contents: list = []

        if image_path and Path(image_path).exists():
            img_bytes = Path(image_path).read_bytes()
            suffix = Path(image_path).suffix.lower().lstrip(".")
            mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

        if video_path and Path(video_path).exists():
            vid_bytes = Path(video_path).read_bytes()
            suffix = Path(video_path).suffix.lower().lstrip(".")
            mime = f"video/{suffix}" if suffix != "mov" else "video/quicktime"
            contents.append(types.Part.from_bytes(data=vid_bytes, mime_type=mime))

        contents.append(structured_prompt)

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
        )

        text = response.text or ""
        confidence = self._extract_confidence(text)

        # Strip the confidence JSON from the response text
        clean_text = re.sub(
            r'\s*\{"confidence":\s*[\d.]+\}\s*$', "", text
        ).strip()

        return clean_text, confidence

    async def judge(
        self,
        prompt: str,
        response: str,
        expected: str,
    ) -> tuple[float, str]:
        """Use Gemini as an LLM judge to score a response.

        Args:
            prompt: The original task prompt.
            response: The model's response to evaluate.
            expected: The expected/reference answer.

        Returns:
            A tuple of (score 0-1, explanation).
        """
        judge_prompt = (
            "You are an expert evaluator. Score the following response against "
            "the expected answer.\n\n"
            f"TASK PROMPT: {prompt}\n\n"
            f"MODEL RESPONSE: {response}\n\n"
            f"EXPECTED ANSWER: {expected}\n\n"
            "Evaluate how well the response captures the key elements of the "
            "expected answer. Consider correctness, completeness, and relevance.\n\n"
            "Respond with ONLY a JSON object:\n"
            '{{"score": <float 0.0 to 1.0>, "explanation": "<brief explanation>"}}'
        )

        result = self.client.models.generate_content(
            model=self.model_id,
            contents=[judge_prompt],
        )

        text = result.text or ""
        return self._parse_judge_response(text)

    async def health_check(self) -> bool:
        """Verify the Gemini API is reachable with a simple test query.

        Returns:
            True if Gemini responds successfully, False otherwise.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=["Respond with exactly: OK"],
            )
            return bool(response.text)
        except Exception:
            return False

    @staticmethod
    def _extract_confidence(text: str) -> float:
        """Extract confidence score from model response text.

        Args:
            text: The raw model response that may contain a confidence JSON.

        Returns:
            The extracted confidence value, or 0.5 as a default.
        """
        match = re.search(r'\{"confidence":\s*([\d.]+)\}', text)
        if match:
            try:
                value = float(match.group(1))
                return max(0.0, min(1.0, value))
            except ValueError:
                pass
        return 0.5

    @staticmethod
    def _parse_judge_response(text: str) -> tuple[float, str]:
        """Parse a judge response JSON into score and explanation.

        Args:
            text: The raw judge response text.

        Returns:
            A tuple of (score, explanation).
        """
        # Try to find JSON in the response
        json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0.5))
                explanation = str(data.get("explanation", "No explanation provided."))
                return max(0.0, min(1.0, score)), explanation
            except (json.JSONDecodeError, ValueError):
                pass
        return 0.5, f"Could not parse judge response: {text[:200]}"
