"""Tests for vacancy_parser.config — isolated from the real .env and shell env."""

from pathlib import Path

import pytest

from vacancy_parser.config import ConfigError, load_settings

_DEFAULT_MODEL = "gemini-3.5-flash-lite"
_TEST_KEY = "test-secret-key-do-not-leak"
_PADDED_SECRET = "  sk-padded-secret-should-not-leak  "


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
    message = str(exc_info.value)
    assert "GEMINI_API_KEY" in message
    assert "not set or empty" in message
    assert "whitespace" not in message


def test_padded_key_error_hides_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    # Non-empty padded key fails validation; hide_input_in_errors must keep the
    # raw value out of ConfigError and the chained ValidationError.
    monkeypatch.setenv("GEMINI_API_KEY", _PADDED_SECRET)
    with pytest.raises(ConfigError) as exc_info:
        load_settings(env_file=None)
    err = exc_info.value
    message = str(err)
    assert "GEMINI_API_KEY" in message
    assert "whitespace" in message
    assert "not set or empty" not in message
    assert _PADDED_SECRET not in message
    assert "sk-padded-secret-should-not-leak" not in message
    assert err.__cause__ is not None
    assert _PADDED_SECRET not in str(err.__cause__)
    assert "sk-padded-secret-should-not-leak" not in str(err.__cause__)


def test_secret_not_in_repr_or_str(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", _TEST_KEY)
    settings = load_settings(env_file=None)
    assert _TEST_KEY not in repr(settings)
    assert _TEST_KEY not in str(settings)


def test_reads_values_from_dotenv_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "GEMINI_API_KEY=from-dotenv-file\nGEMINI_MODEL=model-from-file\n",
        encoding="utf-8",
    )
    settings = load_settings(env_file=env_path)
    assert settings.gemini_api_key.get_secret_value() == "from-dotenv-file"
    assert settings.gemini_model == "model-from-file"


def test_env_var_overrides_dotenv_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "GEMINI_API_KEY=from-dotenv-file\nGEMINI_MODEL=model-from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell-env")
    monkeypatch.setenv("GEMINI_MODEL", "model-from-shell")
    settings = load_settings(env_file=env_path)
    assert settings.gemini_api_key.get_secret_value() == "from-shell-env"
    assert settings.gemini_model == "model-from-shell"


def test_empty_key_in_dotenv_raises_config_error(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_settings(env_file=env_path)
    assert "GEMINI_API_KEY" in str(exc_info.value)
