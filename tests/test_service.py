"""Tests for vacancy parse service: validation retry, errors, and attempt metadata."""

import json

import pytest

from vacancy_parser.llm import FakeLLMClient, LLMClient, LLMError, LLMResponse
from vacancy_parser.schemas import JobPosting
from vacancy_parser.service import ParseError, parse_vacancy

VACANCY_TEXT = "Senior Angular Developer, fully remote, B2B."


def _response(
    text: str,
    *,
    model: str = "fake-model",
    input_tokens: int = 10,
    output_tokens: int = 5,
    latency_ms: float = 1.5,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )


def test_success_on_first_attempt(valid_job_dict: dict[str, object]) -> None:
    payload = json.dumps(valid_job_dict)
    fake = FakeLLMClient([_response(payload)])
    # Annotated use site: pyright checks FakeLLMClient against LLMClient.
    client: LLMClient = fake

    result = parse_vacancy(VACANCY_TEXT, client)

    assert result.posting.title == "Senior Angular Developer"
    assert result.posting.seniority.value == "senior"
    assert len(fake.prompts) == 1
    assert fake.schemas == [JobPosting]
    assert len(result.attempts) == 1
    assert VACANCY_TEXT in fake.prompts[0]


def test_broken_json_then_valid_retries_once(valid_job_dict: dict[str, object]) -> None:
    valid = json.dumps(valid_job_dict)
    client = FakeLLMClient(
        [
            _response("{not-json"),
            _response(valid),
        ]
    )

    result = parse_vacancy(VACANCY_TEXT, client)

    assert result.posting.title == "Senior Angular Developer"
    assert len(client.prompts) == 2
    assert len(result.attempts) == 2
    retry_prompt = client.prompts[1]
    assert VACANCY_TEXT in retry_prompt
    # Broken JSON surfaces as a validation error; essence must reach the model.
    assert "json" in retry_prompt.casefold() or "parse" in retry_prompt.casefold()


def test_schema_violation_then_valid_mentions_salary(
    valid_job_dict: dict[str, object],
) -> None:
    invalid = {
        **valid_job_dict,
        "salary": {
            "min": 200,
            "max": 100,
            "currency": "PLN",
            "period": "hour",
        },
    }
    client = FakeLLMClient(
        [
            _response(json.dumps(invalid)),
            _response(json.dumps(valid_job_dict)),
        ]
    )

    result = parse_vacancy(VACANCY_TEXT, client)

    assert result.posting.salary is None
    assert len(client.prompts) == 2
    assert "salary" in client.prompts[1].casefold()
    assert VACANCY_TEXT in client.prompts[1]


def test_invalid_both_attempts_raises_parse_error(
    valid_job_dict: dict[str, object],
) -> None:
    invalid = {**valid_job_dict, "title": ""}
    client = FakeLLMClient(
        [
            _response(json.dumps(invalid)),
            _response(json.dumps(invalid)),
        ]
    )

    with pytest.raises(ParseError, match="2 attempts"):
        parse_vacancy(VACANCY_TEXT, client)

    assert len(client.prompts) == 2


def test_llm_error_propagates_without_retry() -> None:
    client = FakeLLMClient([LLMError("upstream 429")])

    with pytest.raises(LLMError, match="upstream 429"):
        parse_vacancy(VACANCY_TEXT, client)

    assert len(client.prompts) == 1


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_or_whitespace_input_makes_no_llm_call(text: str) -> None:
    client = FakeLLMClient([_response("{}")])

    with pytest.raises(ValueError, match="empty"):
        parse_vacancy(text, client)

    assert client.prompts == []
    assert client.schemas == []


def test_attempt_metadata_includes_both_tries(valid_job_dict: dict[str, object]) -> None:
    client = FakeLLMClient(
        [
            _response(
                "{broken",
                input_tokens=11,
                output_tokens=2,
                latency_ms=10.0,
            ),
            _response(
                json.dumps(valid_job_dict),
                input_tokens=22,
                output_tokens=33,
                latency_ms=40.0,
            ),
        ]
    )

    result = parse_vacancy(VACANCY_TEXT, client)

    assert len(result.attempts) == 2
    assert result.attempts[0].input_tokens == 11
    assert result.attempts[0].output_tokens == 2
    assert result.attempts[0].latency_ms == 10.0
    assert result.attempts[1].input_tokens == 22
    assert result.attempts[1].output_tokens == 33
    assert result.attempts[1].latency_ms == 40.0


def test_retry_prompt_omits_pydantic_noise(valid_job_dict: dict[str, object]) -> None:
    invalid = {
        **valid_job_dict,
        "salary": {
            "min": 200,
            "max": 100,
            "currency": "PLN",
            "period": "hour",
        },
    }
    client = FakeLLMClient(
        [
            _response(json.dumps(invalid)),
            _response(json.dumps(valid_job_dict)),
        ]
    )

    parse_vacancy(VACANCY_TEXT, client)

    retry_prompt = client.prompts[1]
    assert "input_value" not in retry_prompt
    assert "errors.pydantic.dev" not in retry_prompt
    assert "https://" not in retry_prompt
