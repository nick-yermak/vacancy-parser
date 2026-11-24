"""Parse vacancy text into a validated JobPosting via an injected LLM client."""

from dataclasses import dataclass

from pydantic import ValidationError

from vacancy_parser.llm import LLMClient, LLMResponse
from vacancy_parser.schemas import JobPosting

MAX_ATTEMPTS = 2

PROMPT_TEMPLATE = """\
Extract structured job posting data from the vacancy text below.
Return JSON that matches the provided schema. Do not invent facts that are not stated.

Vacancy text:
{vacancy_text}
"""

RETRY_PROMPT_TEMPLATE = """\
Extract structured job posting data from the vacancy text below.
Return JSON that matches the provided schema. Do not invent facts that are not stated.

Your previous response failed validation. Fix these errors:
{errors}

Vacancy text:
{vacancy_text}
"""


class ParseError(Exception):
    """Raised when the model response cannot be validated after all attempts."""


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Validated posting plus metadata from every LLM attempt."""

    posting: JobPosting
    attempts: tuple[LLMResponse, ...]


def parse_vacancy(text: str, client: LLMClient) -> ParseResult:
    """Validate vacancy text via the LLM client, with one schema-repair retry.

    Raises
    ------
    ValueError
        Empty or whitespace-only ``text`` (no LLM call).
    LLMError
        Propagated from the client without retry.
    ParseError
        Validation failed on every attempt.
    """
    if not text.strip():
        raise ValueError("vacancy text is empty")

    attempts: list[LLMResponse] = []
    errors_text: str | None = None

    for _ in range(MAX_ATTEMPTS):
        if errors_text is None:
            prompt = PROMPT_TEMPLATE.format(vacancy_text=text)
        else:
            prompt = RETRY_PROMPT_TEMPLATE.format(
                vacancy_text=text,
                errors=errors_text,
            )

        # LLMError from the client propagates: no validation retry on transport failures.
        response = client.complete(prompt, JobPosting)
        attempts.append(response)

        try:
            # One path for broken JSON and schema violations (no separate json.loads).
            posting = JobPosting.model_validate_json(response.text)
        except ValidationError as exc:
            errors_text = _format_validation_errors(exc)
            continue

        return ParseResult(posting=posting, attempts=tuple(attempts))

    raise ParseError(f"failed to parse vacancy after {MAX_ATTEMPTS} attempts: {errors_text}")


def _format_validation_errors(exc: ValidationError) -> str:
    """Short field/message lines for the retry prompt — not ``str(exc)``."""
    lines: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        msg = error["msg"]
        lines.append(f"{loc}: {msg}" if loc else msg)
    return "\n".join(lines)
