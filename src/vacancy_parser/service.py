"""Parse vacancy text into a validated JobPosting via an injected LLM client."""

from dataclasses import dataclass

from pydantic import ValidationError

from vacancy_parser.llm import LLMClient, LLMError, LLMResponse
from vacancy_parser.schemas import JobPosting

MAX_ATTEMPTS = 2
MAX_VALIDATION_ERRORS = 5

PROMPT_INSTRUCTION = """\
Extract structured job posting data from the vacancy text below.
Return JSON that matches the provided schema. Do not invent facts that are not stated.\
"""

RETRY_ERRORS_TEMPLATE = """\
Your previous response failed validation. Fix these errors:
{errors}\
"""

VACANCY_SECTION_TEMPLATE = """\
Vacancy text:
{vacancy_text}\
"""


class EmptyVacancyError(Exception):
    """Raised when vacancy text is empty or whitespace-only (no LLM call)."""


class ParseError(Exception):
    """Raised when the model response cannot be validated after all attempts."""

    def __init__(self, message: str, *, attempts: tuple[LLMResponse, ...]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Validated posting plus metadata from every LLM attempt."""

    posting: JobPosting
    attempts: tuple[LLMResponse, ...]


def parse_vacancy(text: str, client: LLMClient) -> ParseResult:
    """Validate vacancy text via the LLM client, with one schema-repair retry.

    Raises
    ------
    EmptyVacancyError
        Empty or whitespace-only ``text`` (no LLM call).
    LLMError
        Propagated from the client without retry. Prior paid attempts in this
        invocation are attached as ``attempts`` (empty on first-call failure).
    ParseError
        Validation failed on every attempt (``attempts`` holds call metadata).
    """
    if not text.strip():
        raise EmptyVacancyError("vacancy text is empty")

    attempts: list[LLMResponse] = []
    errors_text: str | None = None

    for _ in range(MAX_ATTEMPTS):
        prompt = _build_prompt(text, errors=errors_text)

        try:
            # No validation retry on transport failures; keep prior paid metadata.
            response = client.complete(prompt, JobPosting)
        except LLMError as exc:
            raise LLMError(str(exc), attempts=tuple(attempts)) from exc

        attempts.append(response)

        try:
            # One path for broken JSON and schema violations (no separate json.loads).
            posting = JobPosting.model_validate_json(response.text)
        except ValidationError as exc:
            errors_text = _format_validation_errors(exc)
            continue

        return ParseResult(posting=posting, attempts=tuple(attempts))

    raise ParseError(
        f"failed to parse vacancy after {MAX_ATTEMPTS} attempts: {errors_text}",
        attempts=tuple(attempts),
    )


def _build_prompt(vacancy_text: str, *, errors: str | None = None) -> str:
    """Assemble the LLM prompt; retry inserts the error block into the base shape."""
    parts = [PROMPT_INSTRUCTION]
    if errors is not None:
        parts.append(RETRY_ERRORS_TEMPLATE.format(errors=errors))
    parts.append(VACANCY_SECTION_TEMPLATE.format(vacancy_text=vacancy_text))
    return "\n\n".join(parts) + "\n"


def _format_validation_errors(exc: ValidationError) -> str:
    """Short field/message lines for the retry prompt — not ``str(exc)``."""
    errors = exc.errors()
    lines: list[str] = []
    for error in errors[:MAX_VALIDATION_ERRORS]:
        loc = ".".join(str(part) for part in error["loc"])
        msg = error["msg"]
        lines.append(f"{loc}: {msg}" if loc else msg)
    omitted = len(errors) - MAX_VALIDATION_ERRORS
    if omitted > 0:
        lines.append(f"... and {omitted} more error(s)")
    return "\n".join(lines)
