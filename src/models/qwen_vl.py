"""Hosted Qwen2.5-VL wrapper using HuggingFace Inference API.

Reads HF_TOKEN from the environment. Uses the huggingface_hub InferenceClient
for chat completions with optional image support.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from huggingface_hub import InferenceClient

from src.models.base import BaseModel

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"


class QwenVLModel(BaseModel):
    """Wrapper for hosted Qwen2.5-VL via HuggingFace Inference API."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        api_key: str | None = None,
    ) -> None:
        """Initialize the Qwen VL client.

        Args:
            model_id: HuggingFace model ID.
            api_key: HF token. Falls back to HF_TOKEN env var.

        Raises:
            ValueError: If no API key is available.
        """
        token = api_key or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "HF_TOKEN environment variable is required. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        self._client = InferenceClient(model=model_id, token=token)
        self.model_id = model_id
        self.name = "qwen2.5-72b"

    async def query(
        self,
        prompt: str,
        image_path: str | None = None,
        video_path: str | None = None,
    ) -> tuple[str, float]:
        """Send a prompt with optional image to hosted Qwen2.5-VL.

        Args:
            prompt: The text prompt to send.
            image_path: Optional path to an image file.
            video_path: Optional path to a video file (noted in prompt only).

        Returns:
            A tuple of (response_text, confidence_score).
        """
        structured_prompt = (
            f"{prompt}\n\n"
            "After your answer, on a new line, provide your confidence as a JSON "
            'object: {{"confidence": <float 0-1>}}'
        )

        if video_path and not image_path:
            structured_prompt = (
                f"[Note: This task involves a video at {video_path}. "
                "Describe what you can infer from the scenario.]\n\n"
                + structured_prompt
            )

        content: list[dict] = []

        if image_path and Path(image_path).exists():
            img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
            suffix = Path(image_path).suffix.lower().lstrip(".")
            mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_b64}"},
            })

        content.append({"type": "text", "text": structured_prompt})

        response = self._client.chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
        )

        text = response.choices[0].message.content or ""
        confidence = self._extract_confidence(text)
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
        """Use Qwen as an LLM judge to score a response.

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

        text, _ = await self.query(judge_prompt)
        return self._parse_judge_response(text)

    async def health_check(self) -> bool:
        """Ping the hosted endpoint to verify it is reachable.

        Returns:
            True if the endpoint responds, False otherwise.
        """
        try:
            response, _ = await self.query("Say hello.")
            return len(response) > 0
        except Exception:
            return False

    @staticmethod
    def _extract_confidence(text: str) -> float:
        """Extract confidence score from model response text."""
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
        """Parse a judge response JSON into score and explanation."""
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
