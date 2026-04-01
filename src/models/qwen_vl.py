"""Hosted Qwen2.5-VL wrapper supporting HuggingFace Inference API and Together AI.

Reads QWEN_API_KEY and QWEN_API_BASE from the environment. Falls back to
HuggingFace Inference API if QWEN_API_BASE is not set.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import httpx

from src.models.base import BaseModel

# Default endpoints for each provider
HF_INFERENCE_BASE = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-VL-7B-Instruct"
TOGETHER_AI_BASE = "https://api.together.xyz/v1/chat/completions"
TOGETHER_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


class QwenVLModel(BaseModel):
    """Wrapper for hosted Qwen2.5-VL via HuggingFace Inference API or Together AI."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        """Initialize the Qwen VL client.

        Args:
            api_key: API key. Falls back to QWEN_API_KEY env var.
            api_base: Base URL for the API. Falls back to QWEN_API_BASE env var,
                      then to HuggingFace Inference API.

        Raises:
            ValueError: If no API key is available.
        """
        self.api_key = api_key or os.environ.get("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "QWEN_API_KEY environment variable is required. "
                "Set it to your HuggingFace or Together AI API key."
            )
        self.api_base = api_base or os.environ.get("QWEN_API_BASE", "")
        self._is_together = "together" in self.api_base.lower()
        self._is_hf = not self._is_together

        if not self.api_base:
            self.api_base = HF_INFERENCE_BASE
            self._is_hf = True
            self._is_together = False

        self.name = "qwen2.5-vl-7b"
        self._client = httpx.AsyncClient(timeout=120.0)

    async def query(
        self,
        prompt: str,
        image_path: str | None = None,
        video_path: str | None = None,
    ) -> tuple[str, float]:
        """Send a prompt with optional image to the hosted Qwen endpoint.

        Note: Video is not directly supported by most hosted endpoints.
        If a video path is provided, only the first frame concept is described
        in the prompt.

        Args:
            prompt: The text prompt to send.
            image_path: Optional path to an image file.
            video_path: Optional path to a video file (limited support).

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

        if self._is_together:
            text = await self._query_together(structured_prompt, image_path)
        else:
            text = await self._query_hf(structured_prompt, image_path)

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

    async def _query_hf(self, prompt: str, image_path: str | None) -> str:
        """Send a query to the HuggingFace Inference API.

        Args:
            prompt: The text prompt.
            image_path: Optional image file path.

        Returns:
            The response text from the model.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}

        content: list[dict] = []
        if image_path and Path(image_path).exists():
            img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
            suffix = Path(image_path).suffix.lower().lstrip(".")
            mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_b64}"},
            })

        content.append({"type": "text", "text": prompt})

        payload = {
            "inputs": "",
            "parameters": {
                "messages": [{"role": "user", "content": content}],
                "max_new_tokens": 1024,
            },
        }

        resp = await self._client.post(
            self.api_base, headers=headers, json=payload
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "")
        if isinstance(data, dict):
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return data.get("generated_text", "")
        return str(data)

    async def _query_together(self, prompt: str, image_path: str | None) -> str:
        """Send a query to the Together AI API.

        Args:
            prompt: The text prompt.
            image_path: Optional image file path.

        Returns:
            The response text from the model.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        content: list[dict] = []
        if image_path and Path(image_path).exists():
            img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
            suffix = Path(image_path).suffix.lower().lstrip(".")
            mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_b64}"},
            })

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": TOGETHER_MODEL_ID,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1024,
        }

        resp = await self._client.post(
            self.api_base, headers=headers, json=payload
        )
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

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
