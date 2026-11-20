import pytest


@pytest.fixture
def valid_job_dict() -> dict[str, object]:
    """Minimal valid JobPosting payload; tests copy and break one field."""
    return {
        "title": "Senior Angular Developer",
        "company": None,
        "seniority": "senior",
        "remote": "remote",
        "contract_types": [],
        "tech_stack": [],
        "salary": None,
        "languages": [],
        "red_flags": [],
        "summary": "Remote Angular role on B2B.",
    }


@pytest.fixture
def full_job_dict(valid_job_dict: dict[str, object]) -> dict[str, object]:
    """Fully populated valid JobPosting payload."""
    return {
        **valid_job_dict,
        "company": "Acme Sp. z o.o.",
        "contract_types": ["b2b", "uop"],
        "tech_stack": ["Angular", "TypeScript", "RxJS"],
        "salary": {
            "min": 120,
            "max": 160,
            "currency": "PLN",
            "period": "hour",
        },
        "languages": [
            {"language": "English", "level": "b2"},
            {"language": "Polish", "level": "c1"},
        ],
        "red_flags": ["unpaid overtime expected"],
        "summary": "Senior Angular IC, fully remote, B2B from 120 PLN/h.",
    }
