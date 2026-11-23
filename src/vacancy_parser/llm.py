"""LLM client boundary: protocol, response metadata, errors, and a fake client.

Validation of the model JSON into JobPosting (or any schema) is intentionally
left to the service layer — see module decisions in the feat(llm) commit.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


class LLMError(Exception):
    """Raised when the LLM API call fails. SDK exceptions must not leak past llm.py."""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Raw model text plus call metadata for logging."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient(Protocol):
    """Structural client seam. Checked by the type checker, not isinstance."""

    def complete[T: BaseModel](
        self,
        prompt: str,
        response_schema: type[T],
    ) -> LLMResponse:
        """Send prompt with a JSON response schema; return raw text + metadata."""
        ...


class FakeLLMClient:
    """Scripted client for tests: queued responses/errors and recorded prompts."""

    def __init__(self, script: Sequence[LLMResponse | BaseException]) -> None:
        self._script = list(script)
        self.prompts: list[str] = []

    def complete[T: BaseModel](
        self,
        prompt: str,
        response_schema: type[T],
    ) -> LLMResponse:
        del response_schema  # accepted for Protocol parity; unused by the fake
        self.prompts.append(prompt)
        if not self._script:
            raise RuntimeError("FakeLLMClient script exhausted: no responses left")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item
