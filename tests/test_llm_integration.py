"""Live Gemini integration tests (skipped by default via addopts)."""

from __future__ import annotations

import pytest

from vacancy_parser.config import ConfigError, load_settings
from vacancy_parser.llm import GeminiLLMClient
from vacancy_parser.schemas import JobPosting
from vacancy_parser.service import parse_vacancy

SHORT_VACANCY = (
    "Acme is hiring a Senior Angular Developer. Fully remote, B2B only. "
    "Stack: Angular, TypeScript, RxJS. English B2."
)


@pytest.mark.integration
def test_parse_vacancy_with_real_gemini_client() -> None:
    try:
        settings = load_settings()
    except ConfigError as exc:
        pytest.skip(f"Gemini credentials unavailable: {exc}")

    client: GeminiLLMClient
    client = GeminiLLMClient(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_model,
    )

    result = parse_vacancy(SHORT_VACANCY, client)

    assert isinstance(result.posting, JobPosting)
    assert result.posting.title
    assert len(result.attempts) >= 1
    assert result.attempts[0].input_tokens is not None
    assert result.attempts[0].input_tokens > 0
    assert result.attempts[0].output_tokens is not None
    assert result.attempts[0].output_tokens > 0
