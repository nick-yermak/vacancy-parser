"""LLM client boundary: protocol, Gemini client, fake, response metadata, errors.

Gemini schema notes (JobPosting.model_json_schema → response_json_schema)
--------------------------------------------------------------------------
``$defs``/``$ref`` and ``anyOf``-null (required-nullable fields) are accepted.
``maxLength``/``minLength`` on strings (e.g. summary ≤ 300) are stripped/ignored
by constrained decoding — service validation catches oversize text. Currency has
no JSON Schema ``pattern`` (Pydantic validator only). Very large/deep schemas may
be rejected with 400.

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
L — generateContent for phase 0. Single-turn JSON extraction matches
    ``models.generate_content`` + ``response_mime_type`` /
    ``response_json_schema``; Interactions is the better fit for agentic /
    multi-turn work in phase 1. generateContent remains fully supported in
    google-genai 3.6.1; migrating now would add API churn without product gain.
M — Token counts are ``int | None``. Missing usage_metadata must not become 0 —
    step-7 cost logs would understate spend; None means unknown, not free.
N — One LLMError with ``status`` and ``retryable``. Enough for logging and
    transport-retry without an isinstance zoo; subclasses remain possible.
    ``attempts`` is service-layer metadata the client never sets; parse_vacancy
    attaches prior paid attempts onto the same exception instance so type and
    fields survive.
O — Own retry loop (3 attempts, exponential backoff, injectable sleep). SDK
    HttpRetryOptions stay unset (no tenacity retries) so we do not double-retry
    and can assert attempt counts without real sleeps. Retry 408/429/5xx and
    network/timeout; never retry 400/401/403 or empty/blocked responses.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel

TRANSPORT_ATTEMPTS = 3
RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_INITIAL_BACKOFF_S = 0.5


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Raw model text plus call metadata for logging."""

    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float


class LLMError(Exception):
    """Raised when the LLM API call fails. SDK exceptions must not leak past llm.py.

    ``status`` is an HTTP code when known (None for empty responses / network).
    ``retryable`` marks transport failures the client already retried or would.
    ``attempts`` carries metadata from earlier successful ``complete`` calls in the
    same service invocation (e.g. a paid response that failed validation before a
    later transport error). Empty when the failure is on the first call; the
    client never sets this field.
    """

    def __init__(
        self,
        message: str = "",
        *,
        status: int | None = None,
        retryable: bool = False,
        attempts: tuple[LLMResponse, ...] = (),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
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


class GeminiLLMClient:
    """google-genai wrapper. Sole import site for ``google.genai`` in this package."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        transport_attempts: int = TRANSPORT_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
        _sdk_client: Any | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._transport_attempts = transport_attempts
        self._sleep = sleep
        # Testing seam: inject a fake client; production builds Client from the key.
        self._client = _sdk_client if _sdk_client is not None else genai.Client(api_key=api_key)

    def __repr__(self) -> str:
        return (
            f"GeminiLLMClient(model={self._model!r}, transport_attempts={self._transport_attempts})"
        )

    def complete(
        self,
        prompt: str,
        response_schema: type[BaseModel],
    ) -> LLMResponse:
        """Call generateContent with native JSON schema; map failures to LLMError."""
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=response_schema.model_json_schema(),
        )
        last_error: LLMError | None = None

        for attempt_index in range(self._transport_attempts):
            started = time.monotonic()
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
            except Exception as exc:
                mapped = self._map_exception(exc)
                last_error = mapped
                if not mapped.retryable or attempt_index + 1 >= self._transport_attempts:
                    raise mapped from exc
                self._sleep(_INITIAL_BACKOFF_S * (2**attempt_index))
                continue

            # Empty/blocked responses are not transport failures — no retry.
            return self._to_llm_response(response, started=started)

        assert last_error is not None  # pragma: no cover - loop always sets or returns
        raise last_error

    def _to_llm_response(
        self,
        response: genai_types.GenerateContentResponse,
        *,
        started: float,
    ) -> LLMResponse:
        text = response.text
        finish_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason

        block_reason = None
        if response.prompt_feedback is not None:
            block_reason = response.prompt_feedback.block_reason

        if text is None or text == "":
            detail_parts = []
            if finish_reason is not None:
                detail_parts.append(f"finish_reason={finish_reason}")
            if block_reason is not None:
                detail_parts.append(f"block_reason={block_reason}")
            if not response.candidates:
                detail_parts.append("no candidates")
            detail = ", ".join(detail_parts) if detail_parts else "empty text"
            raise LLMError(
                f"empty model response ({detail})",
                status=None,
                retryable=False,
            )

        if finish_reason is not None and finish_reason != genai_types.FinishReason.STOP:
            raise LLMError(
                f"model stopped early (finish_reason={finish_reason})",
                status=None,
                retryable=False,
            )

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage is not None else None
        output_tokens = usage.candidates_token_count if usage is not None else None
        model = response.model_version or self._model
        latency_ms = (time.monotonic() - started) * 1000.0

        return LLMResponse(
            text=text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    def _map_exception(self, exc: BaseException) -> LLMError:
        message = self._safe_message(exc)
        if isinstance(exc, genai_errors.APIError):
            code = exc.code
            retryable = code in RETRYABLE_HTTP_STATUS or (500 <= code < 600)
            return LLMError(message, status=code, retryable=retryable)
        # Network / timeout / unexpected SDK failures: treat as retryable transport.
        return LLMError(message, status=None, retryable=True)

    def _safe_message(self, exc: BaseException) -> str:
        raw = str(exc)
        if self._api_key and self._api_key in raw:
            raw = raw.replace(self._api_key, "***")
        return raw or exc.__class__.__name__
