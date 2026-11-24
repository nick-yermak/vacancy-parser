"""Tests for FakeLLMClient and GeminiLLMClient (no network in the default run)."""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel

from vacancy_parser.llm import (
    TRANSPORT_ATTEMPTS,
    FakeLLMClient,
    GeminiLLMClient,
    LLMClient,
    LLMError,
    LLMResponse,
)
from vacancy_parser.schemas import JobPosting

API_KEY = "test-secret-key-do-not-leak"


class _TinySchema(BaseModel):
    value: str


class _FakeModels:
    def __init__(self, script: list[Any]) -> None:
        self._script: deque[Any] = deque(script)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._script:
            raise RuntimeError("fake SDK script exhausted")
        item = self._script.popleft()
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeSdkClient:
    def __init__(self, script: list[Any]) -> None:
        self.models = _FakeModels(script)


def _response(text: str, *, model: str = "fake-model") -> LLMResponse:
    return LLMResponse(
        text=text,
        model=model,
        input_tokens=10,
        output_tokens=5,
        latency_ms=1.5,
    )


def _sdk_response(
    text: str | None,
    *,
    finish_reason: genai_types.FinishReason | None = genai_types.FinishReason.STOP,
    prompt_tokens: int | None = 11,
    candidate_tokens: int | None = 7,
    model_version: str | None = "gemini-fake-version",
    candidates: list[genai_types.Candidate] | None = None,
    block_reason: genai_types.BlockedReason | None = None,
) -> genai_types.GenerateContentResponse:
    if candidates is None and text is not None:
        candidates = [
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[genai_types.Part(text=text)],
                    role="model",
                ),
                finish_reason=finish_reason,
            )
        ]
    usage = None
    if prompt_tokens is not None or candidate_tokens is not None:
        usage = genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidate_tokens,
        )
    prompt_feedback = None
    if block_reason is not None:
        prompt_feedback = genai_types.GenerateContentResponsePromptFeedback(
            block_reason=block_reason,
        )
    return genai_types.GenerateContentResponse(
        candidates=candidates,
        model_version=model_version,
        usage_metadata=usage,
        prompt_feedback=prompt_feedback,
    )


def _client(
    script: list[Any],
    *,
    sleep: Any = None,
    transport_attempts: int = TRANSPORT_ATTEMPTS,
) -> tuple[GeminiLLMClient, _FakeSdkClient]:
    def _noop_sleep(_seconds: float) -> None:
        pass

    fake = _FakeSdkClient(script)
    client = GeminiLLMClient(
        api_key=API_KEY,
        model="gemini-test",
        transport_attempts=transport_attempts,
        sleep=sleep or _noop_sleep,
        _sdk_client=fake,
    )
    return client, fake


def test_fake_satisfies_llm_client_protocol() -> None:
    # Protocol is checked at use sites; this assignment is the static seam check.
    client: LLMClient = FakeLLMClient([_response("ok")])
    assert client.complete("p", _TinySchema).text == "ok"


def test_gemini_satisfies_llm_client_protocol() -> None:
    # Annotated assignment: pyright checks GeminiLLMClient against LLMClient.
    client, _fake = _client([_sdk_response('{"value":"x"}')])
    typed: LLMClient = client
    assert typed.complete("p", _TinySchema).text == '{"value":"x"}'


def test_fake_returns_scripted_responses_in_order() -> None:
    client = FakeLLMClient(
        [
            _response("{not-json"),
            _response("second"),
        ]
    )
    assert client.complete("p1", _TinySchema).text == "{not-json"
    assert client.complete("p2", JobPosting).text == "second"


def test_fake_records_prompts_and_schemas_in_order() -> None:
    client = FakeLLMClient([_response("a"), _response("b")])
    client.complete("prompt-one", _TinySchema)
    client.complete("prompt-two", JobPosting)
    assert client.prompts == ["prompt-one", "prompt-two"]
    assert client.schemas == [_TinySchema, JobPosting]


def test_fake_raises_scripted_exception() -> None:
    client = FakeLLMClient([LLMError("upstream failed")])
    with pytest.raises(LLMError, match="upstream failed"):
        client.complete("p", _TinySchema)
    assert client.prompts == ["p"]
    assert client.schemas == [_TinySchema]


def test_fake_exhausted_script_raises_clear_error() -> None:
    client = FakeLLMClient([_response("only")])
    client.complete("p1", _TinySchema)
    with pytest.raises(RuntimeError, match="script exhausted"):
        client.complete("p2", JobPosting)
    assert client.prompts == ["p1", "p2"]
    assert client.schemas == [_TinySchema, JobPosting]


def test_gemini_maps_response_to_llm_response() -> None:
    client, fake = _client([_sdk_response('{"value":"ok"}')])

    result = client.complete("extract", _TinySchema)

    assert result.text == '{"value":"ok"}'
    assert result.model == "gemini-fake-version"
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert result.latency_ms >= 0.0
    call = fake.models.calls[0]
    assert call["model"] == "gemini-test"
    assert call["contents"] == "extract"
    config = call["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == _TinySchema.model_json_schema()


def test_gemini_usage_without_tokens_is_none() -> None:
    response = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[genai_types.Part(text='{"value":"ok"}')],
                    role="model",
                ),
                finish_reason=genai_types.FinishReason.STOP,
            )
        ],
        model_version="gemini-fake-version",
        usage_metadata=None,
    )
    client, _fake = _client([response])

    result = client.complete("p", _TinySchema)

    assert result.input_tokens is None
    assert result.output_tokens is None


def test_gemini_empty_response_raises_llm_error() -> None:
    client, _fake = _client(
        [
            _sdk_response(
                None,
                candidates=[],
                block_reason=genai_types.BlockedReason.SAFETY,
            )
        ]
    )

    with pytest.raises(LLMError, match="empty model response") as exc_info:
        client.complete("p", _TinySchema)

    assert exc_info.value.retryable is False
    assert exc_info.value.status is None


def test_gemini_non_stop_finish_reason_raises() -> None:
    client, _fake = _client(
        [
            _sdk_response(
                '{"value":"partial"}',
                finish_reason=genai_types.FinishReason.MAX_TOKENS,
            )
        ]
    )

    with pytest.raises(LLMError, match="finish_reason") as exc_info:
        client.complete("p", _TinySchema)

    assert exc_info.value.retryable is False


@pytest.mark.parametrize(
    ("exc", "status", "retryable"),
    [
        (
            genai_errors.ClientError(
                429,
                {"error": {"message": "rate limited", "status": "RESOURCE_EXHAUSTED"}},
            ),
            429,
            True,
        ),
        (
            genai_errors.ServerError(
                503,
                {"error": {"message": "unavailable", "status": "UNAVAILABLE"}},
            ),
            503,
            True,
        ),
        (
            genai_errors.ClientError(
                400,
                {"error": {"message": "bad request", "status": "INVALID_ARGUMENT"}},
            ),
            400,
            False,
        ),
        (ConnectionError(f"connect failed using {API_KEY}"), None, True),
    ],
)
def test_gemini_maps_errors(
    exc: BaseException,
    status: int | None,
    retryable: bool,
) -> None:
    # One attempt: assert mapping only (retry coverage lives in dedicated tests).
    client, _fake = _client([exc], transport_attempts=1)

    with pytest.raises(LLMError) as exc_info:
        client.complete("p", _TinySchema)

    assert exc_info.value.status == status
    assert exc_info.value.retryable is retryable
    assert API_KEY not in str(exc_info.value)
    assert API_KEY not in repr(client)
    assert exc_info.value.__cause__ is exc


def test_gemini_retries_retryable_errors_without_real_sleep() -> None:
    sleeps: list[float] = []
    client, fake = _client(
        [
            genai_errors.ClientError(
                429,
                {"error": {"message": "rate limited", "status": "RESOURCE_EXHAUSTED"}},
            ),
            genai_errors.ServerError(
                500,
                {"error": {"message": "boom", "status": "INTERNAL"}},
            ),
            _sdk_response('{"value":"ok"}'),
        ],
        sleep=sleeps.append,
    )

    result = client.complete("p", _TinySchema)

    assert result.text == '{"value":"ok"}'
    assert len(fake.models.calls) == 3
    assert len(sleeps) == 2
    assert sleeps == [0.5, 1.0]


def test_gemini_gives_up_after_transport_attempts() -> None:
    sleeps: list[float] = []
    errors = [
        genai_errors.ClientError(
            429,
            {"error": {"message": f"rate {i}", "status": "RESOURCE_EXHAUSTED"}},
        )
        for i in range(TRANSPORT_ATTEMPTS)
    ]
    client, fake = _client(errors, sleep=sleeps.append)

    with pytest.raises(LLMError) as exc_info:
        client.complete("p", _TinySchema)

    assert exc_info.value.status == 429
    assert exc_info.value.retryable is True
    assert len(fake.models.calls) == TRANSPORT_ATTEMPTS
    assert len(sleeps) == TRANSPORT_ATTEMPTS - 1


def test_gemini_does_not_retry_client_errors() -> None:
    sleeps: list[float] = []
    client, fake = _client(
        [
            genai_errors.ClientError(
                401,
                {"error": {"message": "unauthorized", "status": "UNAUTHENTICATED"}},
            ),
            _sdk_response('{"value":"ok"}'),
        ],
        sleep=sleeps.append,
    )

    with pytest.raises(LLMError) as exc_info:
        client.complete("p", _TinySchema)

    assert exc_info.value.status == 401
    assert exc_info.value.retryable is False
    assert len(fake.models.calls) == 1
    assert sleeps == []


def test_gemini_does_not_retry_empty_response() -> None:
    sleeps: list[float] = []
    client, fake = _client(
        [
            _sdk_response(None, candidates=[]),
            _sdk_response('{"value":"ok"}'),
        ],
        sleep=sleeps.append,
    )

    with pytest.raises(LLMError, match="empty model response"):
        client.complete("p", _TinySchema)

    assert len(fake.models.calls) == 1
    assert sleeps == []


def test_gemini_sdk_client_injection_avoids_real_client_when_provided() -> None:
    # Guard: constructing with an injected client must not need a live network.
    sdk = _FakeSdkClient([_sdk_response('{"value":"x"}')])
    client = GeminiLLMClient(api_key=API_KEY, model="m", _sdk_client=sdk)

    assert client.complete("p", _TinySchema).text == '{"value":"x"}'
    assert len(sdk.models.calls) == 1
