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
    Empty/MAX_TOKENS responses still bill tokens: usage is attached as
    ``LLMError.partial`` so step 7 can count them.
N — One LLMError with ``status`` and ``retryable``. Enough for logging and
    transport-retry without an isinstance zoo; subclasses remain possible.
    ``attempts`` is service-layer metadata the client never sets; parse_vacancy
    attaches prior paid attempts onto the same exception instance so type and
    fields survive.
O — Own retry loop (injectable sleep). Prefer Gemini ``RetryInfo.retryDelay``;
    fall back to 15s for 429 (free-tier RPM) and 2s for 5xx/network. Do not
    retry PerDay quota 429s, 400/401/403, or empty/blocked. SDK HttpRetryOptions
    stay unset. ``transport_calls`` on LLMResponse/LLMError exposes HTTP tries
    (including retries) for step-7 RPD logs. ``__cause__`` may still hold a raw
    SDK error — CLI must not print tracebacks (step 6).
"""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel

TRANSPORT_ATTEMPTS = 3
RETRYABLE_HTTP_STATUS = frozenset({408, 429})
_DEFAULT_429_BACKOFF_S = 15.0
_DEFAULT_TRANSPORT_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 60.0
_RETRY_DELAY_RE = re.compile(r"(?:retry(?:Delay)?|retry in)\D*([0-9]+(?:\.[0-9]+)?)\s*s", re.I)


def _is_transport_exception(exc: BaseException) -> bool:
    """SDK/HTTP failures only — programming errors (TypeError, …) must propagate."""
    if isinstance(exc, genai_errors.APIError):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    # google-genai 3.6.1 may ship httpx2 instead of httpx; match either by module.
    return type(exc).__module__.startswith("httpx")


class _GenerativeModels(Protocol):
    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: genai_types.GenerateContentConfig,
    ) -> genai_types.GenerateContentResponse: ...


class _GenaiClient(Protocol):
    @property
    def models(self) -> _GenerativeModels: ...


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Raw model text plus call metadata for logging."""

    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    transport_calls: int = 1


class LLMError(Exception):
    """Raised when the LLM API call fails. SDK exceptions must not leak past llm.py.

    ``status`` is an HTTP code when known (None for empty responses / network).
    ``retryable`` marks transport failures the client already retried or would.
    ``transport_calls`` is how many HTTP requests this ``complete`` made.
    ``partial`` holds usage metadata when the API responded but content was
    unusable (empty / non-STOP); None for pure transport failures.
    ``attempts`` carries metadata from earlier successful ``complete`` calls in the
    same service invocation. Empty when the failure is on the first call; the
    client never sets this field.
    """

    def __init__(
        self,
        message: str = "",
        *,
        status: int | None = None,
        retryable: bool = False,
        transport_calls: int = 1,
        partial: LLMResponse | None = None,
        attempts: tuple[LLMResponse, ...] = (),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.transport_calls = transport_calls
        self.partial = partial
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
        _sdk_client: _GenaiClient | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._transport_attempts = transport_attempts
        self._sleep = sleep
        # Testing seam: inject a fake client; production builds Client from the key.
        self._client: _GenaiClient = (
            _sdk_client if _sdk_client is not None else genai.Client(api_key=api_key)
        )

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
        attempt_index = 0
        while True:
            started = time.monotonic()
            transport_calls = attempt_index + 1
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
            except Exception as exc:
                if not _is_transport_exception(exc):
                    raise
                mapped = self._map_exception(exc, transport_calls=transport_calls)
                attempt_index += 1
                if mapped.retryable and attempt_index < self._transport_attempts:
                    self._sleep(
                        self._backoff_seconds(
                            exc,
                            mapped,
                            attempt_index=attempt_index - 1,
                        )
                    )
                    continue
                if self._api_key and self._api_key in str(exc):
                    raise mapped from None
                raise mapped from exc

            # Empty/blocked responses are not transport failures — no retry.
            return self._to_llm_response(
                response,
                started=started,
                transport_calls=transport_calls,
            )

    def _to_llm_response(
        self,
        response: genai_types.GenerateContentResponse,
        *,
        started: float,
        transport_calls: int,
    ) -> LLMResponse:
        text = response.text
        finish_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason

        block_reason = None
        if response.prompt_feedback is not None:
            block_reason = response.prompt_feedback.block_reason

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage is not None else None
        output_tokens = usage.candidates_token_count if usage is not None else None
        model = response.model_version or self._model
        latency_ms = (time.monotonic() - started) * 1000.0

        if text is None or text == "":
            detail_parts = []
            if finish_reason is not None:
                detail_parts.append(f"finish_reason={finish_reason}")
            if block_reason is not None:
                detail_parts.append(f"block_reason={block_reason}")
            if not response.candidates:
                detail_parts.append("no candidates")
            detail = ", ".join(detail_parts) if detail_parts else "empty text"
            partial = LLMResponse(
                text="",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                transport_calls=transport_calls,
            )
            raise LLMError(
                f"empty model response ({detail})",
                status=None,
                retryable=False,
                transport_calls=transport_calls,
                partial=partial,
            )

        if finish_reason is not None and finish_reason != genai_types.FinishReason.STOP:
            partial = LLMResponse(
                text=text,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                transport_calls=transport_calls,
            )
            raise LLMError(
                f"model stopped early (finish_reason={finish_reason})",
                status=None,
                retryable=False,
                transport_calls=transport_calls,
                partial=partial,
            )

        return LLMResponse(
            text=text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            transport_calls=transport_calls,
        )

    def _map_exception(self, exc: BaseException, *, transport_calls: int) -> LLMError:
        message = self._safe_message(exc)
        if isinstance(exc, genai_errors.APIError):
            code = exc.code
            retryable = self._is_retryable_api_error(exc)
            return LLMError(
                message,
                status=code,
                retryable=retryable,
                transport_calls=transport_calls,
            )
        return LLMError(
            message,
            status=None,
            retryable=True,
            transport_calls=transport_calls,
        )

    def _is_retryable_api_error(self, exc: genai_errors.APIError) -> bool:
        code = exc.code
        if code in RETRYABLE_HTTP_STATUS or 500 <= code < 600:
            # Daily quota 429s will not clear within our retry window.
            return not (code == 429 and self._is_per_day_quota(exc))
        return False

    def _backoff_seconds(
        self,
        exc: BaseException,
        error: LLMError,
        *,
        attempt_index: int,
    ) -> float:
        if error.status == 429 and isinstance(exc, genai_errors.APIError):
            delay = self._retry_delay_seconds(exc)
            if delay is None:
                delay = _DEFAULT_429_BACKOFF_S * (2**attempt_index)
            return min(delay, _MAX_BACKOFF_S)
        delay = _DEFAULT_TRANSPORT_BACKOFF_S * (2**attempt_index)
        return min(delay, _MAX_BACKOFF_S)

    def _safe_message(self, exc: BaseException) -> str:
        raw = str(exc)
        if self._api_key and self._api_key in raw:
            raw = raw.replace(self._api_key, "***")
        return raw or exc.__class__.__name__

    @staticmethod
    def _error_details_list(exc: genai_errors.APIError) -> list[object]:
        payload = exc.details
        if isinstance(payload, dict):
            error = payload.get("error", payload)
            if isinstance(error, dict):
                details = error.get("details", [])
                if isinstance(details, list):
                    return details
        return []

    @classmethod
    def _is_per_day_quota(cls, exc: genai_errors.APIError) -> bool:
        for item in cls._error_details_list(exc):
            if not isinstance(item, dict):
                continue
            violations = item.get("violations", [])
            if not isinstance(violations, list):
                continue
            for violation in violations:
                if not isinstance(violation, dict):
                    continue
                quota_id = str(violation.get("quotaId", ""))
                if "PerDay" in quota_id:
                    return True
        return False

    @classmethod
    def _retry_delay_seconds(cls, exc: genai_errors.APIError) -> float | None:
        for item in cls._error_details_list(exc):
            if not isinstance(item, dict):
                continue
            raw = item.get("retryDelay")
            if isinstance(raw, str):
                match = _RETRY_DELAY_RE.search(f"retryDelay {raw}")
                if match:
                    return float(match.group(1))
            if isinstance(raw, (int, float)):
                return float(raw)
        match = _RETRY_DELAY_RE.search(str(exc))
        if match:
            return float(match.group(1))
        return None
