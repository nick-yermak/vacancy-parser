"""LLM client boundary: protocol, response metadata, errors, and a fake client.

Design decisions
----------------
F — Service validates. The client returns raw response text plus call metadata;
    JobPosting (or any schema) parsing and retry live above this module. That
    lets FakeLLMClient script invalid JSON without a client-side validator.
G — complete() takes type[BaseModel], not a TypeVar. A type parameter would only
    pay off if the return type depended on the schema (e.g. client returns T).
    Under F it does not — the client always returns LLMResponse. The service also
    keeps a concrete JobPosting in ParseResult; a generic belongs only if a
    caller-chosen schema drove the return type.
H — FakeLLMClient lives here (not in tests/) so service tests can inject it
    without importing from tests/, and so it stays available for dry-run/demo.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Raw model text plus call metadata for logging."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMError(Exception):
    """Raised when the LLM API call fails. SDK exceptions must not leak past llm.py.

    ``attempts`` carries metadata from earlier successful ``complete`` calls in the
    same service invocation (e.g. a paid response that failed validation before a
    later transport error). Empty when the failure is on the first call.
    """

    def __init__(
        self,
        message: str = "",
        *,
        attempts: tuple[LLMResponse, ...] = (),
    ) -> None:
        super().__init__(message)
        self.attempts = attempts


class LLMClient(Protocol):
    """Structural client seam. Checked by the type checker, not isinstance."""

    def complete(
        self,
        prompt: str,
        response_schema: type[BaseModel],
    ) -> LLMResponse:
        """Send prompt with a JSON response schema; return raw text + metadata."""
        ...


class FakeLLMClient:
    """Scripted client for tests: queued responses/errors and recorded calls."""

    def __init__(self, script: Sequence[LLMResponse | BaseException]) -> None:
        self._script: deque[LLMResponse | BaseException] = deque(script)
        self.prompts: list[str] = []
        self.schemas: list[type[BaseModel]] = []

    def complete(
        self,
        prompt: str,
        response_schema: type[BaseModel],
    ) -> LLMResponse:
        self.prompts.append(prompt)
        self.schemas.append(response_schema)
        if not self._script:
            raise RuntimeError("FakeLLMClient script exhausted: no responses left")
        item = self._script.popleft()
        if isinstance(item, BaseException):
            raise item
        return item
