"""CLI tests via Typer CliRunner — no network, env isolated from the real .env."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from vacancy_parser import cli
from vacancy_parser.cli import EXIT_OK, EXIT_RUNTIME, EXIT_USER, app, map_cli_error
from vacancy_parser.config import ConfigError
from vacancy_parser.llm import FakeLLMClient, LLMClient, LLMError, LLMResponse
from vacancy_parser.schemas import JobPosting
from vacancy_parser.service import EmptyVacancyError, ParseError

_TEST_KEY = "test-cli-secret-key-do-not-leak"
_PADDED_KEY = "  padded-cli-secret-must-not-appear  "
VACANCY_TEXT = "Senior Angular Developer, fully remote, B2B."

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Avoid the developer's .env / shell secrets; cwd is an empty temp dir."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


def _response(text: str, *, model: str = "fake-model") -> LLMResponse:
    return LLMResponse(
        text=text,
        model=model,
        input_tokens=10,
        output_tokens=5,
        latency_ms=1.0,
    )


def _install_fake(
    monkeypatch: pytest.MonkeyPatch,
    script: list[LLMResponse | BaseException],
    *,
    api_key: str = _TEST_KEY,
    created: dict[str, str] | None = None,
) -> FakeLLMClient:
    monkeypatch.setenv("GEMINI_API_KEY", api_key)
    fake = FakeLLMClient(script)
    sink: dict[str, str] = created if created is not None else {}

    def factory(*, api_key: str, model: str) -> FakeLLMClient:
        sink["api_key"] = api_key
        sink["model"] = model
        return fake

    monkeypatch.setattr(cli, "build_llm_client", factory)
    return fake


def _assert_no_traceback_or_secret(result: Result) -> None:
    for blob in (result.output, result.stdout, result.stderr):
        assert "Traceback" not in blob
        assert _TEST_KEY not in blob
        assert "padded-cli-secret-must-not-appear" not in blob


def test_parse_file_stdout_is_pure_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_job_dict: dict[str, object],
) -> None:
    vacancy = tmp_path / "vacancy.txt"
    vacancy.write_text(VACANCY_TEXT, encoding="utf-8")
    _install_fake(monkeypatch, [_response(json.dumps(valid_job_dict))])

    result = runner.invoke(app, ["parse", str(vacancy)])

    assert result.exit_code == 0
    posting = JobPosting.model_validate_json(result.stdout)
    assert posting.title == "Senior Angular Developer"
    assert result.stderr == ""
    _assert_no_traceback_or_secret(result)


def test_parse_output_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_job_dict: dict[str, object],
) -> None:
    vacancy = tmp_path / "vacancy.txt"
    vacancy.write_text(VACANCY_TEXT, encoding="utf-8")
    out = tmp_path / "nested" / "out.json"
    _install_fake(monkeypatch, [_response(json.dumps(valid_job_dict))])

    result = runner.invoke(app, ["parse", str(vacancy), "-o", str(out)])

    assert result.exit_code == 0
    assert result.stdout == ""
    posting = JobPosting.model_validate_json(out.read_text(encoding="utf-8"))
    assert posting.title == "Senior Angular Developer"
    _assert_no_traceback_or_secret(result)


def test_parse_output_overwrites_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_job_dict: dict[str, object],
) -> None:
    vacancy = tmp_path / "vacancy.txt"
    vacancy.write_text(VACANCY_TEXT, encoding="utf-8")
    out = tmp_path / "out.json"
    out.write_text("stale-content-should-be-gone", encoding="utf-8")
    _install_fake(monkeypatch, [_response(json.dumps(valid_job_dict))])

    result = runner.invoke(app, ["parse", str(vacancy), "-o", str(out)])

    assert result.exit_code == 0
    text = out.read_text(encoding="utf-8")
    assert "stale-content-should-be-gone" not in text
    JobPosting.model_validate_json(text)
    _assert_no_traceback_or_secret(result)


def test_parse_stdin_dash(
    monkeypatch: pytest.MonkeyPatch,
    valid_job_dict: dict[str, object],
) -> None:
    _install_fake(monkeypatch, [_response(json.dumps(valid_job_dict))])

    result = runner.invoke(app, ["parse", "-"], input=VACANCY_TEXT)

    assert result.exit_code == 0
    posting = JobPosting.model_validate_json(result.stdout)
    assert posting.company is None
    _assert_no_traceback_or_secret(result)


def test_missing_file_exit_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", _TEST_KEY)
    result = runner.invoke(app, ["parse", "no-such-vacancy.txt"])

    assert result.exit_code == EXIT_USER
    assert "file not found" in result.stderr.casefold()
    assert result.stdout == ""
    _assert_no_traceback_or_secret(result)


def test_empty_file_does_not_call_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vacancy = tmp_path / "empty.txt"
    vacancy.write_text("   \n", encoding="utf-8")
    fake = _install_fake(monkeypatch, [LLMError("should not be called")])

    result = runner.invoke(app, ["parse", str(vacancy)])

    assert result.exit_code == EXIT_USER
    assert fake.prompts == []
    assert "empty" in result.stderr.casefold()
    _assert_no_traceback_or_secret(result)


def test_empty_stdin_without_api_key_exit_1() -> None:
    """Empty input must be exit 1 even when GEMINI_API_KEY is unset."""
    result = runner.invoke(app, ["parse", "-"], input="  \n")

    assert result.exit_code == EXIT_USER
    assert "empty" in result.stderr.casefold()
    assert "GEMINI_API_KEY" not in result.stderr
    _assert_no_traceback_or_secret(result)


def test_non_utf8_file_exit_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vacancy = tmp_path / "binary.txt"
    vacancy.write_bytes(b"\xff\xfe\x00not-utf8")
    monkeypatch.setenv("GEMINI_API_KEY", _TEST_KEY)

    result = runner.invoke(app, ["parse", str(vacancy)])

    assert result.exit_code == EXIT_USER
    assert "utf-8" in result.stderr.casefold()
    _assert_no_traceback_or_secret(result)


def test_output_directory_checked_before_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_job_dict: dict[str, object],
) -> None:
    vacancy = tmp_path / "vacancy.txt"
    vacancy.write_text(VACANCY_TEXT, encoding="utf-8")
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    fake = _install_fake(monkeypatch, [_response(json.dumps(valid_job_dict))])

    result = runner.invoke(app, ["parse", str(vacancy), "-o", str(out_dir)])

    assert result.exit_code == EXIT_USER
    assert "output path is a directory" in result.stderr
    assert fake.prompts == []
    _assert_no_traceback_or_secret(result)


def test_parse_error_exit_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vacancy = tmp_path / "vacancy.txt"
    vacancy.write_text(VACANCY_TEXT, encoding="utf-8")
    _install_fake(
        monkeypatch,
        [_response("{not-json"), _response("{still-bad")],
    )

    result = runner.invoke(app, ["parse", str(vacancy)])

    assert result.exit_code == EXIT_USER
    assert result.stdout == ""
    assert result.stderr.strip() != ""
    _assert_no_traceback_or_secret(result)


def test_llm_error_exit_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vacancy = tmp_path / "vacancy.txt"
    vacancy.write_text(VACANCY_TEXT, encoding="utf-8")
    _install_fake(
        monkeypatch,
        [LLMError("upstream 503", status=503, retryable=True)],
    )

    result = runner.invoke(app, ["parse", str(vacancy)])

    assert result.exit_code == EXIT_RUNTIME
    assert result.stderr.strip() == "upstream 503"
    _assert_no_traceback_or_secret(result)


def test_unexpected_client_error_exit_2_no_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vacancy = tmp_path / "vacancy.txt"
    vacancy.write_text(VACANCY_TEXT, encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", _TEST_KEY)

    leak = f"https://example.com/?key={_TEST_KEY}"

    class _BoomClient:
        def complete(self, prompt: str, response_schema: type) -> LLMResponse:
            raise RuntimeError(f"sdk failed: {leak}")

    def factory(*, api_key: str, model: str) -> LLMClient:
        return _BoomClient()  # type: ignore[return-value]

    monkeypatch.setattr(cli, "build_llm_client", factory)

    result = runner.invoke(app, ["parse", str(vacancy)])

    assert result.exit_code == EXIT_RUNTIME
    assert result.stderr.strip() == "unexpected error (RuntimeError)"
    assert leak not in result.stderr
    assert leak not in result.output
    assert "Traceback" not in result.output
    assert "Traceback" not in result.stderr
    _assert_no_traceback_or_secret(result)


def test_broken_pipe_exits_quietly_without_io_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_job_dict: dict[str, object],
) -> None:
    vacancy = tmp_path / "vacancy.txt"
    vacancy.write_text(VACANCY_TEXT, encoding="utf-8")
    _install_fake(monkeypatch, [_response(json.dumps(valid_job_dict))])

    def _raise_broken_pipe(_text: str) -> None:
        raise BrokenPipeError()

    monkeypatch.setattr(cli, "_write_stdout", _raise_broken_pipe)

    result = runner.invoke(app, ["parse", str(vacancy)])

    assert result.exit_code == EXIT_OK
    assert "I/O error" not in result.stderr
    assert "BrokenPipe" not in result.stderr
    assert "Exception ignored" not in result.output
    _assert_no_traceback_or_secret(result)


def test_config_error_exit_2_hides_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", _PADDED_KEY)
    Path("vacancy.txt").write_text(VACANCY_TEXT, encoding="utf-8")

    result = runner.invoke(app, ["parse", "vacancy.txt"])

    assert result.exit_code == EXIT_RUNTIME
    assert "GEMINI_API_KEY" in result.stderr
    assert _PADDED_KEY not in result.stderr
    assert "padded-cli-secret-must-not-appear" not in result.stderr
    assert "padded-cli-secret-must-not-appear" not in result.stdout
    assert "padded-cli-secret-must-not-appear" not in result.output
    _assert_no_traceback_or_secret(result)


def test_model_option_reaches_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_job_dict: dict[str, object],
) -> None:
    vacancy = tmp_path / "vacancy.txt"
    vacancy.write_text(VACANCY_TEXT, encoding="utf-8")
    created: dict[str, str] = {}
    _install_fake(
        monkeypatch,
        [_response(json.dumps(valid_job_dict))],
        created=created,
    )

    result = runner.invoke(
        app,
        ["parse", str(vacancy), "--model", "gemini-custom-for-cli"],
    )

    assert result.exit_code == 0
    assert created["model"] == "gemini-custom-for-cli"
    assert created["api_key"] == _TEST_KEY
    _assert_no_traceback_or_secret(result)


def test_default_model_from_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_job_dict: dict[str, object],
) -> None:
    vacancy = tmp_path / "vacancy.txt"
    vacancy.write_text(VACANCY_TEXT, encoding="utf-8")
    monkeypatch.setenv("GEMINI_MODEL", "model-from-settings")
    created: dict[str, str] = {}
    _install_fake(
        monkeypatch,
        [_response(json.dumps(valid_job_dict))],
        created=created,
    )

    result = runner.invoke(app, ["parse", str(vacancy)])

    assert result.exit_code == 0
    assert created["model"] == "model-from-settings"
    _assert_no_traceback_or_secret(result)


def test_help_does_not_expose_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", _TEST_KEY)
    result = runner.invoke(app, ["parse", "--help"])

    assert result.exit_code == 0
    assert _TEST_KEY not in result.stdout
    assert _TEST_KEY not in result.stderr
    assert "Traceback" not in result.output
    assert "Exit codes:" in result.stdout


def test_map_cli_error_known_and_unexpected_codes() -> None:
    assert map_cli_error(EmptyVacancyError("vacancy text is empty"))[1] == EXIT_USER
    assert map_cli_error(ParseError("boom", attempts=()))[1] == EXIT_USER
    assert map_cli_error(LLMError("api down"))[1] == EXIT_RUNTIME
    assert map_cli_error(ConfigError("missing key"))[1] == EXIT_RUNTIME
    assert map_cli_error(FileNotFoundError(2, "nope", "x.txt"))[1] == EXIT_USER
    message, code = map_cli_error(RuntimeError("secret-in-message"))
    assert code == EXIT_RUNTIME
    assert message == "unexpected error (RuntimeError)"
    assert "secret-in-message" not in message
