"""Abstract base class for model wrappers.

All model integrations (Gemini, Qwen, etc.) implement this interface so the
evaluation harness and observer can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract interface that every model wrapper must implement."""

    name: str

    @abstractmethod
    async def query(
        self,
        prompt: str,
        image_path: str | None = None,
        video_path: str | None = None,
    ) -> tuple[str, float]:
        """Send a prompt with optional media to the model.

        Args:
            prompt: The text prompt to send.
            image_path: Optional path to an image file.
            video_path: Optional path to a video file.

        Returns:
            A tuple of (response_text, confidence_score) where confidence is 0-1.
        """
        ...

    @abstractmethod
    async def judge(
        self,
        prompt: str,
        response: str,
        expected: str,
    ) -> tuple[float, str]:
        """Score a response against an expected answer using the model as judge.

        Args:
            prompt: The original task prompt.
            response: The model's response to evaluate.
            expected: The expected/reference answer.

        Returns:
            A tuple of (score 0-1, explanation).
        """
        ...

    async def health_check(self) -> bool:
        """Verify the model endpoint is reachable.

        Returns:
            True if the model responds successfully, False otherwise.
        """
        try:
            response, _ = await self.query("Say hello.")
            return len(response) > 0
        except Exception:
            return False
