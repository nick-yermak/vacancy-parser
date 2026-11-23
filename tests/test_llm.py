"""Tests for FakeLLMClient scripting and prompt recording."""

import pytest
from pydantic import BaseModel

from vacancy_parser.llm import FakeLLMClient, LLMError, LLMResponse
from vacancy_parser.schemas import JobPosting


class _TinySchema(BaseModel):
    value: str


def _response(text: str, *, model: str = "fake-model") -> LLMResponse:
    return LLMResponse(
        text=text,
        model=model,
        input_tokens=10,
        output_tokens=5,
        latency_ms=1.5,
    )


def test_fake_returns_scripted_responses_in_order() -> None:
    client = FakeLLMClient(
        [
            _response("first"),
            _response("second"),
        ]
    )
    assert client.complete("p1", _TinySchema).text == "first"
    assert client.complete("p2", JobPosting).text == "second"


def test_fake_records_prompts_in_order() -> None:
    client = FakeLLMClient([_response("a"), _response("b")])
    client.complete("prompt-one", _TinySchema)
    client.complete("prompt-two", _TinySchema)
    assert client.prompts == ["prompt-one", "prompt-two"]


def test_fake_raises_scripted_exception() -> None:
    client = FakeLLMClient([LLMError("upstream failed")])
    with pytest.raises(LLMError, match="upstream failed"):
        client.complete("p", _TinySchema)
    assert client.prompts == ["p"]


def test_fake_exhausted_script_raises_clear_error() -> None:
    client = FakeLLMClient([_response("only")])
    client.complete("p1", _TinySchema)
    with pytest.raises(RuntimeError, match="script exhausted"):
        client.complete("p2", _TinySchema)
    assert client.prompts == ["p1", "p2"]


def test_fake_can_return_invalid_json_for_service_retry_tests() -> None:
    # Decision F: client returns raw text; service validates — fake can script junk.
    client = FakeLLMClient([_response("{not-json"), _response('{"value":"ok"}')])
    assert client.complete("p", _TinySchema).text == "{not-json"
