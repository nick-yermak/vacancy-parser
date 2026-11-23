import json
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


def _resolve_def(schema: dict[str, object], name: str) -> dict[str, object]:
    defs = schema.get("$defs") or schema.get("definitions")
    assert isinstance(defs, dict), "expected $defs/definitions in JSON schema"
    resolved = defs[name]
    assert isinstance(resolved, dict)
    return cast(dict[str, object], resolved)


def _valid_salary(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "min": 20_000,
        "max": None,
        "currency": "PLN",
        "period": "month",
    }
    data.update(overrides)
    return data


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
    def test_missing_company_key_rejected(self, valid_job_dict: dict[str, object]) -> None:
        del valid_job_dict["company"]
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(valid_job_dict)
        _assert_loc(exc_info.value, "company")

    def test_missing_salary_key_rejected(self, valid_job_dict: dict[str, object]) -> None:
        del valid_job_dict["salary"]
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(valid_job_dict)
        _assert_loc(exc_info.value, "salary")

    @pytest.mark.parametrize("field", ["min", "max", "currency", "period"])
    def test_salary_missing_nested_key_rejected(self, field: str) -> None:
        salary = _valid_salary()
        del salary[field]
        with pytest.raises(ValidationError) as exc_info:
            Salary.model_validate(salary)
        _assert_loc(exc_info.value, field)

    def test_language_missing_level_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            LanguageRequirement.model_validate({"language": "English"})
        _assert_loc(exc_info.value, "level")

    def test_salary_required_keys_in_json_schema(self) -> None:
        schema = JobPosting.model_json_schema()
        salary = _resolve_def(schema, "Salary")
        assert set(cast(list[str], salary["required"])) == {
            "min",
            "max",
            "currency",
            "period",
        }

    def test_blank_company_becomes_none(self, valid_job_dict: dict[str, object]) -> None:
        valid_job_dict["company"] = "  "
        posting = JobPosting.model_validate(valid_job_dict)
        assert posting.company is None

    def test_empty_title_rejected(self, valid_job_dict: dict[str, object]) -> None:
        valid_job_dict["title"] = "   "
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(valid_job_dict)
        _assert_loc(exc_info.value, "title")

    def test_title_strips_whitespace(self, valid_job_dict: dict[str, object]) -> None:
        valid_job_dict["title"] = "  Senior Dev  "
        posting = JobPosting.model_validate(valid_job_dict)
        assert posting.title == "Senior Dev"


class TestEnums:
    def test_unknown_remote_value_rejected(self, valid_job_dict: dict[str, object]) -> None:
        valid_job_dict["remote"] = "remote-first"
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(valid_job_dict)
        _assert_loc(exc_info.value, "remote")

    def test_enum_case_sensitive_no_normalization(self, valid_job_dict: dict[str, object]) -> None:
        # Decision B: StrEnum values are lowercase; "Remote" is rejected, not coerced.
        valid_job_dict["remote"] = "Remote"
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(valid_job_dict)
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
        valid_job_dict[field] = value
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(valid_job_dict)
        locs = _error_locs(exc_info.value)
        assert any(loc and loc[0] == field for loc in locs), locs


class TestSalary:
    def test_min_equals_max_ok(self) -> None:
        salary = Salary.model_validate(
            _valid_salary(min=20_000, max=20_000, currency="EUR", period="hour")
        )
        assert salary.min == salary.max

    def test_min_only_ok(self) -> None:
        # Decision A: max/currency/period keys still required; max may be null.
        salary = Salary.model_validate(_valid_salary(min=20_000, max=None))
        assert salary.max is None
        assert salary.period == SalaryPeriod.MONTH

    def test_max_only_ok(self) -> None:
        salary = Salary.model_validate(_valid_salary(min=None, max=25_000))
        assert salary.min is None

    def test_currency_null_with_amount_ok(self) -> None:
        salary = Salary.model_validate(_valid_salary(currency=None))
        assert salary.currency is None
        assert salary.min == 20_000

    def test_min_greater_than_max_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Salary.model_validate(_valid_salary(min=30_000, max=20_000))
        assert _error_locs(exc_info.value) == [()]

    def test_both_null_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Salary.model_validate(_valid_salary(min=None, max=None))
        assert () in _error_locs(exc_info.value)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Salary.model_validate(_valid_salary(min=-1))
        _assert_loc(exc_info.value, "min")

    @pytest.mark.parametrize(
        "value",
        [
            True,
            "20000.0",
            20000.0,
        ],
    )
    def test_strict_rejects_coerced_amounts(self, value: object) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Salary.model_validate(_valid_salary(min=value))
        _assert_loc(exc_info.value, "min")

    def test_currency_pln_ok(self) -> None:
        salary = Salary.model_validate(_valid_salary(min=100, period="hour"))
        assert salary.currency == "PLN"

    def test_currency_lowercase_normalized(self) -> None:
        # Format fix only: case → UPPER; meaning is not invented.
        salary = Salary.model_validate(_valid_salary(min=100, currency="pln", period="hour"))
        assert salary.currency == "PLN"

    @pytest.mark.parametrize(
        "currency",
        ["PL", "PLNN", "zł", "12A", "$", "ПЛН", "ÄÖÜ"],
    )
    def test_currency_invalid_rejected(self, currency: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Salary.model_validate(_valid_salary(min=100, currency=currency, period="hour"))
        _assert_loc(exc_info.value, "currency")

    def test_invalid_period_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Salary.model_validate(_valid_salary(min=100, period="weekly"))
        _assert_loc(exc_info.value, "period")

    def test_nested_salary_error_loc_on_job(self, valid_job_dict: dict[str, object]) -> None:
        valid_job_dict["salary"] = {
            "min": 200,
            "max": 100,
            "currency": "PLN",
            "period": "hour",
        }
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(valid_job_dict)
        locs = _error_locs(exc_info.value)
        assert any(loc and loc[0] == "salary" for loc in locs), locs


class TestTechStack:
    def test_normalize_dedupe_preserve_order(self, valid_job_dict: dict[str, object]) -> None:
        valid_job_dict["tech_stack"] = [" Angular ", "angular", "RxJS", ""]
        posting = JobPosting.model_validate(valid_job_dict)
        assert posting.tech_stack == ["Angular", "RxJS"]

    def test_empty_list_ok(self, valid_job_dict: dict[str, object]) -> None:
        posting = JobPosting.model_validate(valid_job_dict)
        assert posting.tech_stack == []


class TestLanguages:
    def test_language_requirement_round_trip(self) -> None:
        req = LanguageRequirement(language="English", level=CefrLevel.B2)
        assert req.model_dump(mode="json") == {"language": "English", "level": "b2"}

    def test_dedupe_by_language_keeps_first(self, valid_job_dict: dict[str, object]) -> None:
        valid_job_dict["languages"] = [
            {"language": "English", "level": "b2"},
            {"language": "english", "level": "c1"},
        ]
        posting = JobPosting.model_validate(valid_job_dict)
        assert len(posting.languages) == 1
        assert posting.languages[0].level == CefrLevel.B2

    def test_empty_language_name_rejected(self, valid_job_dict: dict[str, object]) -> None:
        valid_job_dict["languages"] = [{"language": "  ", "level": "b2"}]
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(valid_job_dict)
        locs = _error_locs(exc_info.value)
        assert any("language" in loc for loc in locs), locs


class TestContractTypes:
    def test_dedupe_preserving_order(self, valid_job_dict: dict[str, object]) -> None:
        valid_job_dict["contract_types"] = ["b2b", "uop", "b2b"]
        posting = JobPosting.model_validate(valid_job_dict)
        assert posting.contract_types == [ContractType.B2B, ContractType.UOP]


class TestSummary:
    def test_exactly_300_ok(self, valid_job_dict: dict[str, object]) -> None:
        valid_job_dict["summary"] = "x" * 300
        posting = JobPosting.model_validate(valid_job_dict)
        assert len(posting.summary) == 300

    def test_301_rejected_not_truncated(self, valid_job_dict: dict[str, object]) -> None:
        # Decision D: ValidationError → retry; do not silently truncate.
        valid_job_dict["summary"] = "x" * 301
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(valid_job_dict)
        _assert_loc(exc_info.value, "summary")

    def test_empty_summary_rejected(self, valid_job_dict: dict[str, object]) -> None:
        valid_job_dict["summary"] = ""
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(valid_job_dict)
        _assert_loc(exc_info.value, "summary")


class TestExtraAndRoundTrip:
    def test_extra_key_forbidden(self, valid_job_dict: dict[str, object]) -> None:
        valid_job_dict["invented_field"] = "nope"
        with pytest.raises(ValidationError) as exc_info:
            JobPosting.model_validate(valid_job_dict)
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

        defs = schema.get("$defs") or schema.get("definitions")
        assert isinstance(defs, dict)
        for model_name, model_schema in defs.items():
            assert isinstance(model_schema, dict)
            model_props = model_schema.get("properties")
            if not isinstance(model_props, dict):
                continue
            for field_name, prop in model_props.items():
                assert isinstance(prop, dict)
                assert "description" in prop, f"{model_name}.{field_name} missing description"
