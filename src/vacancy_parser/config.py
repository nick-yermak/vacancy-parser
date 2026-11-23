"""Application settings loaded from environment variables and an optional .env file."""

from pathlib import Path

from pydantic import SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_MODEL = "gemini-3.5-flash-lite"
_DEFAULT_ENV_FILE = ".env"
_MISSING_KEY_MESSAGE = (
    "GEMINI_API_KEY is not set or empty. Copy .env.example to .env and fill in your Gemini API key."
)


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


class Settings(BaseSettings):
    """Runtime settings. Does not print or exit — callers handle failures."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    gemini_api_key: SecretStr
    gemini_model: str = _DEFAULT_MODEL

    @field_validator("gemini_api_key")
    @classmethod
    def reject_blank_or_padded_api_key(cls, value: SecretStr) -> SecretStr:
        # GEMINI_API_KEY= in a copied .env.example is an empty string, not "unset".
        # Reject surrounding whitespace too — do not send " abc " to the API.
        raw = value.get_secret_value()
        if not raw.strip() or raw != raw.strip():
            raise ValueError("must not be empty or have surrounding whitespace")
        return value


def load_settings(*, env_file: str | Path | None = _DEFAULT_ENV_FILE) -> Settings:
    """Load settings from the environment and optional env file.

    Raises ConfigError with a human-readable message on validation failure.
    Does not print or call sys.exit.
    """
    try:
        return Settings(_env_file=env_file)
    except ValidationError as exc:
        raise ConfigError(_humanize_validation_error(exc)) from exc


def _humanize_validation_error(exc: ValidationError) -> str:
    for error in exc.errors():
        loc = error.get("loc", ())
        if loc and loc[0] == "gemini_api_key":
            return _MISSING_KEY_MESSAGE
    return f"Invalid configuration: {exc}"
