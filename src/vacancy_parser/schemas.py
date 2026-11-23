from collections.abc import Callable, Hashable
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Seniority(StrEnum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    UNKNOWN = "unknown"


class RemoteType(StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class ContractType(StrEnum):
    B2B = "b2b"
    UOP = "uop"
    UZ = "uz"
    UOD = "uod"
    OTHER = "other"


class SalaryPeriod(StrEnum):
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    UNKNOWN = "unknown"


class CefrLevel(StrEnum):
    A1 = "a1"
    A2 = "a2"
    B1 = "b1"
    B2 = "b2"
    C1 = "c1"
    C2 = "c2"
    UNKNOWN = "unknown"


def _dedupe_preserve_order[T, H: Hashable](items: list[T], *, key: Callable[[T], H]) -> list[T]:
    """Drop duplicates by ``key``, keeping the first item and insertion order.

    ``set`` is always unordered, so it cannot preserve first-seen spelling or
    stable order. ``dict`` keeps insertion order (guaranteed since 3.7); the
    first-wins loop below is the case-insensitive / custom-key equivalent of
    ``list(dict.fromkeys(items))`` for plain hashables.
    """
    seen: set[H] = set()
    result: list[T] = []
    for item in items:
        k = key(item)
        if k in seen:
            continue
        seen.add(k)
        result.append(item)
    return result


class Salary(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    # Decision A: required-nullable — every key must be present; null/unknown explicit.
    # strict=True on amounts: bool/str coercion (True→1, "20000.0"→20000) is LLM garbage.
    min: int | None = Field(
        ge=0,
        strict=True,
        description=(
            "Lower bound of the salary range as a non-negative integer. "
            "null if only an upper bound is stated (e.g. 'up to 25 000'). "
            "Do not invent a number."
        ),
    )
    max: int | None = Field(
        ge=0,
        strict=True,
        description=(
            "Upper bound of the salary range as a non-negative integer. "
            "null if only a lower bound is stated (e.g. 'from 20 000'). "
            "Do not invent a number."
        ),
    )
    currency: str | None = Field(
        description=(
            "ISO 4217 currency code, exactly 3 ASCII letters, e.g. PLN, EUR, USD. "
            "null if currency is not stated even when an amount is present. "
            "Do not invent; do not use symbols like zł or $."
        ),
    )
    period: SalaryPeriod = Field(
        description=(
            "Pay period for the amounts: hour, day, month, or year. "
            "Use unknown if the posting does not say; do not guess."
        ),
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        # after: typed str|None. Normalize case only; never map symbols → codes.
        if value is None:
            return None
        normalized = value.upper()
        if len(normalized) != 3 or not normalized.isascii() or not normalized.isalpha():
            raise ValueError("currency must be a 3-letter ISO 4217 code (e.g. PLN), not a symbol")
        return normalized

    @model_validator(mode="after")
    def check_range_and_presence(self) -> Self:
        # Cross-field → model_validator(after), not field_validator.
        if self.min is None and self.max is None:
            raise ValueError(
                "salary must include at least min or max; use salary=null when unknown"
            )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("salary.min must be less than or equal to salary.max")
        return self


class LanguageRequirement(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    language: str = Field(
        description=(
            "Language name as written in the posting (e.g. English, Polish). "
            "Do not invent languages that are not mentioned."
        ),
    )
    # Decision A: level is required; emit unknown explicitly when CEFR is absent.
    level: CefrLevel = Field(
        description=(
            "CEFR level if stated explicitly (a1-c2). "
            "Use unknown if the posting does not give a level; do not guess."
        ),
    )

    @field_validator("language")
    @classmethod
    def language_must_be_non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("language must be a non-empty string")
        return value


class JobPosting(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(
        description="Job title exactly as stated in the posting. Do not invent or rephrase.",
    )
    # Decision A: required-nullable — key must be present; null = explicitly unknown.
    company: str | None = Field(
        description=(
            "Hiring company name if stated explicitly; null if not mentioned. "
            "Do not guess the company from context or the domain."
        ),
    )
    seniority: Seniority = Field(
        description=(
            "Seniority level. Allowed: intern, junior, mid, senior, lead, principal, "
            "unknown. For ranges like 'Senior/Lead', pick the lower bound (senior). "
            "Use unknown if the level is not stated; do not guess."
        ),
    )
    remote: RemoteType = Field(
        description=(
            "Work format: remote, hybrid, onsite, or unknown. "
            "Use unknown if not stated explicitly; do not guess."
        ),
    )
    contract_types: list[ContractType] = Field(
        description=(
            "Contract types offered. Polish market: b2b, uop (umowa o pracę), "
            "uz (umowa zlecenie), uod (umowa o dzieło), other. "
            "Empty list if none are stated; do not invent."
        ),
    )
    tech_stack: list[str] = Field(
        description=(
            "Technologies and tools explicitly mentioned. Empty list if none. "
            "Do not invent or expand abbreviations beyond what is written."
        ),
    )
    # Decision A also for salary: required-nullable so LLM must emit null explicitly.
    salary: Salary | None = Field(
        description=(
            "Salary object if any amount is disclosed; null if salary is not stated. "
            "Do not invent figures."
        ),
    )
    languages: list[LanguageRequirement] = Field(
        description=(
            "Required or preferred languages with CEFR level when stated. "
            "Empty list if none are mentioned; do not invent."
        ),
    )
    red_flags: list[str] = Field(
        description=(
            "Candidate-facing concerns: unpaid overtime, vague scope, unpaid trial "
            "tasks, mandatory relocation sold as remote, below-market pay for the "
            "level, gambling/grey industry, etc. Empty list if none."
        ),
    )
    summary: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Concise factual summary of the role, 1-300 characters. "
            "Stay within the limit; do not pad with marketing fluff."
        ),
    )

    @field_validator("title")
    @classmethod
    def title_must_be_non_empty(cls, value: str) -> str:
        # after strip via str_strip_whitespace: blank title is invalid.
        if not value:
            raise ValueError("title must be a non-empty string")
        return value

    @field_validator("company")
    @classmethod
    def empty_company_to_none(cls, value: str | None) -> str | None:
        # Blank string after strip ≡ explicitly absent → null (still required key).
        if value is None or value == "":
            return None
        return value

    @field_validator("contract_types")
    @classmethod
    def dedupe_contract_types(cls, value: list[ContractType]) -> list[ContractType]:
        return _dedupe_preserve_order(value, key=lambda item: item)

    @field_validator("tech_stack")
    @classmethod
    def normalize_tech_stack(cls, value: list[str]) -> list[str]:
        # after: list[str]; items already stripped. Drop empties + casefold dedupe.
        return _dedupe_preserve_order([item for item in value if item], key=str.casefold)

    @field_validator("languages")
    @classmethod
    def dedupe_languages(cls, value: list[LanguageRequirement]) -> list[LanguageRequirement]:
        return _dedupe_preserve_order(value, key=lambda item: item.language.casefold())

    @field_validator("red_flags")
    @classmethod
    def normalize_red_flags(cls, value: list[str]) -> list[str]:
        return _dedupe_preserve_order([item for item in value if item], key=str.casefold)
