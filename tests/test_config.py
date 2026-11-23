"""Tests for vacancy_parser.config — isolated from the real .env and shell env."""

from __future__ import annotations

from pathlib import Path

import pytest

from vacancy_parser.config import ConfigError, load_settings

_DEFAULT_MODEL = "gemini-3.5-flash-lite"
_TEST_KEY = "test-secret-key-do-not-leak"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop real shell / CI secrets so missing-key tests are deterministic."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


def test_key_from_env_uses_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", _TEST_KEY)
    settings = load_settings(env_file=None)
    assert settings.gemini_api_key.get_secret_value() == _TEST_KEY
    assert settings.gemini_model == _DEFAULT_MODEL


def test_model_overridden_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", _TEST_KEY)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-model")
    settings = load_settings(env_file=None)
    assert settings.gemini_model == "gemini-custom-model"


def test_missing_key_raises_config_error() -> None:
    with pytest.raises(ConfigError) as exc_info:
        load_settings(env_file=None)
    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_empty_key_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ConfigError) as exc_info:
        load_settings(env_file=None)
    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_secret_not_in_repr_or_str(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", _TEST_KEY)
    settings = load_settings(env_file=None)
    assert _TEST_KEY not in repr(settings)
    assert _TEST_KEY not in str(settings)


def test_reads_values_from_dotenv_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "GEMINI_API_KEY=from-dotenv-file\nGEMINI_MODEL=model-from-file\n",
        encoding="utf-8",
    )
    # Ensure env vars do not override the file for this case.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    settings = load_settings(env_file=env_path)
    assert settings.gemini_api_key.get_secret_value() == "from-dotenv-file"
    assert settings.gemini_model == "model-from-file"


def test_empty_key_in_dotenv_raises_config_error(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_settings(env_file=env_path)
    assert "GEMINI_API_KEY" in str(exc_info.value)
