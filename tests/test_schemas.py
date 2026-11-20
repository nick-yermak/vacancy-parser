import json
from copy import deepcopy
from typing import cast

import pytest
from pydantic import ValidationError

from vacancy_parser.schemas import (
    CefrLevel,
    ContractType,
    JobPosting,
    LanguageRequirement,
    Salary,
    SalaryPeriod,
)


def _error_locs(exc: ValidationError) -> list[tuple[str | int, ...]]:
    return [tuple(err["loc"]) for err in exc.errors()]


def _assert_loc(exc: ValidationError, *expected: str | int) -> None:
    assert expected in _error_locs(exc), (
        f"expected loc {expected!r} in {_error_locs(exc)!r}; errors={exc.errors()!r}"
    )


class TestValidPayloads:
    def test_minimal_valid(self, valid_job_dict: dict[str, object]) -> None:
        posting = JobPosting.model_validate(valid_job_dict)
        assert posting.company is None
        assert posting.salary is None
        assert posting.contract_types == []
        assert posting.tech_stack == []
        assert posting.languages == []
        assert posting.red_flags == []

    def test_full_valid(self, full_job_dict: dict[str, object]) -> None:
        posting = JobPosting.model_validate(full_job_dict)
        assert posting.company == "Acme Sp. z o.o."
        assert posting.salary is not None
        assert posting.salary.min == 120
        assert posting.languages[0].level == CefrLevel.B2


class TestRequiredNullable:
    def test_missing_company_key_rejected(
        self, valid_job_dict: dict[str, object]
    ) -> None:
        payload = deepcopy(valid_job_dict)
        del payload["company"]
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(payload)
        _assert_loc(exc_info.value, "company")

    def test_missing_salary_key_rejected(
        self, valid_job_dict: dict[str, object]
    ) -> None:
        payload = deepcopy(valid_job_dict)
        del payload["salary"]
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(payload)
        _assert_loc(exc_info.value, "salary")

    def test_blank_company_becomes_none(
        self, valid_job_dict: dict[str, object]
    ) -> None:
        payload = deepcopy(valid_job_dict)
        payload["company"] = "  "
        posting = JobPosting.model_validate(payload)
        assert posting.company is None

    def test_empty_title_rejected(self, valid_job_dict: dict[str, object]) -> None:
        payload = deepcopy(valid_job_dict)
        payload["title"] = "   "
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(payload)
        _assert_loc(exc_info.value, "title")

    def test_title_strips_whitespace(self, valid_job_dict: dict[str, object]) -> None:
        payload = deepcopy(valid_job_dict)
        payload["title"] = "  Senior Dev  "
        posting = JobPosting.model_validate(payload)
        assert posting.title == "Senior Dev"


class TestEnums:
    def test_unknown_remote_value_rejected(
        self, valid_job_dict: dict[str, object]
    ) -> None:
        payload = deepcopy(valid_job_dict)
        payload["remote"] = "remote-first"
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(payload)
        _assert_loc(exc_info.value, "remote")

    def test_enum_case_sensitive_no_normalization(
        self, valid_job_dict: dict[str, object]
    ) -> None:
        # Decision B: StrEnum values are lowercase; "Remote" is rejected, not coerced.
        payload = deepcopy(valid_job_dict)
        payload["remote"] = "Remote"
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(payload)
        _assert_loc(exc_info.value, "remote")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("seniority", "staff"),
            ("seniority", "Senior"),
            ("contract_types", ["B2B"]),
            ("contract_types", ["freelance"]),
        ],
    )
    def test_invalid_enum_values(
        self,
        valid_job_dict: dict[str, object],
        field: str,
        value: object,
    ) -> None:
        payload = deepcopy(valid_job_dict)
        payload[field] = value
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(payload)
        locs = _error_locs(exc_info.value)
        assert any(loc and loc[0] == field for loc in locs), locs


class TestSalary:
    def test_min_equals_max_ok(self) -> None:
        salary = Salary(
            min=20_000, max=20_000, currency="EUR", period=SalaryPeriod.HOUR
        )
        assert salary.min == salary.max

    def test_min_only_ok(self) -> None:
        salary = Salary(min=20_000, currency="PLN", period=SalaryPeriod.MONTH)
        assert salary.max is None

    def test_max_only_ok(self) -> None:
        salary = Salary(max=25_000, currency="PLN", period=SalaryPeriod.MONTH)
        assert salary.min is None

    def test_min_greater_than_max_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Salary(min=30_000, max=20_000, currency="PLN", period=SalaryPeriod.MONTH)
        # model_validator errors land on the model root loc=()
        assert () in _error_locs(exc_info.value) or ("min",) in _error_locs(
            exc_info.value
        )

    def test_both_null_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Salary(currency="PLN", period=SalaryPeriod.MONTH)
        assert () in _error_locs(exc_info.value)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Salary(min=-1, currency="PLN", period=SalaryPeriod.MONTH)
        _assert_loc(exc_info.value, "min")

    def test_currency_pln_ok(self) -> None:
        salary = Salary(min=100, currency="PLN", period=SalaryPeriod.HOUR)
        assert salary.currency == "PLN"

    def test_currency_lowercase_normalized(self) -> None:
        # Format fix only: case → UPPER; meaning is not invented.
        salary = Salary(min=100, currency="pln", period=SalaryPeriod.HOUR)
        assert salary.currency == "PLN"

    @pytest.mark.parametrize("currency", ["PL", "PLNN", "zł", "12A", "$"])
    def test_currency_invalid_rejected(self, currency: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Salary(min=100, currency=currency, period=SalaryPeriod.HOUR)
        _assert_loc(exc_info.value, "currency")

    def test_invalid_period_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Salary.model_validate({"min": 100, "currency": "PLN", "period": "weekly"})
        _assert_loc(exc_info.value, "period")

    def test_nested_salary_error_loc_on_job(
        self, valid_job_dict: dict[str, object]
    ) -> None:
        payload = deepcopy(valid_job_dict)
        payload["salary"] = {
            "min": 200,
            "max": 100,
            "currency": "PLN",
            "period": "hour",
        }
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(payload)
        locs = _error_locs(exc_info.value)
        assert any(loc and loc[0] == "salary" for loc in locs), locs


class TestTechStack:
    def test_normalize_dedupe_preserve_order(
        self, valid_job_dict: dict[str, object]
    ) -> None:
        payload = deepcopy(valid_job_dict)
        payload["tech_stack"] = [" Angular ", "angular", "RxJS", ""]
        posting = JobPosting.model_validate(payload)
        assert posting.tech_stack == ["Angular", "RxJS"]

    def test_empty_list_ok(self, valid_job_dict: dict[str, object]) -> None:
        posting = JobPosting.model_validate(valid_job_dict)
        assert posting.tech_stack == []


class TestLanguages:
    def test_language_requirement_round_trip(self) -> None:
        req = LanguageRequirement(language="English", level=CefrLevel.B2)
        assert req.model_dump(mode="json") == {"language": "English", "level": "b2"}

    def test_dedupe_by_language_keeps_first(
        self, valid_job_dict: dict[str, object]
    ) -> None:
        payload = deepcopy(valid_job_dict)
        payload["languages"] = [
            {"language": "English", "level": "b2"},
            {"language": "english", "level": "c1"},
        ]
        posting = JobPosting.model_validate(payload)
        assert len(posting.languages) == 1
        assert posting.languages[0].level == CefrLevel.B2

    def test_empty_language_name_rejected(
        self, valid_job_dict: dict[str, object]
    ) -> None:
        payload = deepcopy(valid_job_dict)
        payload["languages"] = [{"language": "  ", "level": "b2"}]
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(payload)
        locs = _error_locs(exc_info.value)
        assert any("language" in loc for loc in locs), locs


class TestContractTypes:
    def test_dedupe_preserving_order(self, valid_job_dict: dict[str, object]) -> None:
        payload = deepcopy(valid_job_dict)
        payload["contract_types"] = ["b2b", "uop", "b2b"]
        posting = JobPosting.model_validate(payload)
        assert posting.contract_types == [ContractType.B2B, ContractType.UOP]


class TestSummary:
    def test_exactly_300_ok(self, valid_job_dict: dict[str, object]) -> None:
        payload = deepcopy(valid_job_dict)
        payload["summary"] = "x" * 300
        posting = JobPosting.model_validate(payload)
        assert len(posting.summary) == 300

    def test_301_rejected_not_truncated(
        self, valid_job_dict: dict[str, object]
    ) -> None:
        # Decision D: ValidationError → retry; do not silently truncate.
        payload = deepcopy(valid_job_dict)
        payload["summary"] = "x" * 301
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(payload)
        _assert_loc(exc_info.value, "summary")


class TestExtraAndRoundTrip:
    def test_extra_key_forbidden(self, valid_job_dict: dict[str, object]) -> None:
        payload = deepcopy(valid_job_dict)
        payload["invented_field"] = "nope"
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(payload)
        _assert_loc(exc_info.value, "invented_field")

    def test_json_round_trip(self, full_job_dict: dict[str, object]) -> None:
        original = JobPosting.model_validate(full_job_dict)
        as_json = original.model_dump_json()
        restored = JobPosting.model_validate_json(as_json)
        assert restored == original

        dumped = original.model_dump(mode="json")
        assert dumped["seniority"] == "senior"
        assert dumped["remote"] == "remote"
        assert "Seniority." not in json.dumps(dumped)
        assert dumped["languages"][0]["level"] == "b2"


class TestSchemaDescriptions:
    def test_every_field_has_description(self) -> None:
        schema = JobPosting.model_json_schema()
        props = schema["properties"]
        for name, prop in props.items():
            assert "description" in prop, f"JobPosting.{name} missing description"

        salary_def = _resolve_def(schema, "Salary")
        for name, prop in salary_def["properties"].items():
            assert "description" in prop, f"Salary.{name} missing description"

        language_def = _resolve_def(schema, "LanguageRequirement")
        for name, prop in language_def["properties"].items():
            assert "description" in prop, (
                f"LanguageRequirement.{name} missing description"
            )


def _resolve_def(schema: dict[str, object], name: str) -> dict[str, object]:
    defs = schema.get("$defs") or schema.get("definitions")
    assert isinstance(defs, dict), "expected $defs/definitions in JSON schema"
    resolved = defs[name]
    assert isinstance(resolved, dict)
    return cast(dict[str, object], resolved)
