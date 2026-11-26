"""CLI composition root: read input → settings → LLM client → parse → JSON.

No business logic here — only wiring, I/O, and exit-code mapping.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from vacancy_parser.config import ConfigError, load_settings
from vacancy_parser.llm import GeminiLLMClient, LLMClient, LLMError
from vacancy_parser.service import EmptyVacancyError, ParseError, parse_vacancy

EXIT_OK = 0
EXIT_USER = 1
EXIT_RUNTIME = 2

EXIT_CODES_EPILOG = """\
Exit codes:
  0  success; also quiet exit on broken stdout pipe (e.g. | head)
  1  input/validation (empty text, missing/non-UTF-8 file, ParseError);
     also Ctrl+C → Click prints 'Aborted!' and exits 1
  2  config, LLM/API, or unexpected internal/SDK errors;
     also Click/Typer usage errors (bad argv) exit 2 before our handler —
     usage mistakes are deterministic, so we keep Click's code rather than remap
"""


app = typer.Typer(
    name="vacancy-parser",
    no_args_is_help=True,
    epilog=EXIT_CODES_EPILOG,
    # Locals can hold the API key; never pretty-print them. Disable rich traces
    # entirely — we map known errors ourselves and must not leak a traceback.
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _root() -> None:
    """Parse job vacancy text into structured JobPosting JSON via Gemini.

    Empty callback so ``parse`` stays a subcommand. Without it, Typer collapses a
    single ``@app.command`` into the root (``vacancy-parser <file>``).
    """


class OutputPathError(Exception):
    """Raised when ``-o`` cannot be used; checked before any LLM call."""


def build_llm_client(*, api_key: str, model: str) -> LLMClient:
    """Construct the production Gemini client.

    Tests monkeypatch this factory (Angular-style provider override) so the CLI
    module never grows a ``if testing`` branch.
    """
    return GeminiLLMClient(api_key=api_key, model=model)


def map_cli_error(exc: BaseException) -> tuple[str, int]:
    """Map an exception to ``(stderr message, exit code)`` — single source of truth.

    Exit 1 — caller/input/validation problems.
    Exit 2 — config, LLM/API, or unexpected internal/SDK failures (not the user's
    vacancy text). Unexpected bugs must not look like ``EXIT_USER``.

    ``ParseError.attempts`` / ``LLMError.attempts`` stay on ``exc`` for step-7
    logging; this function only chooses the human message and code.
    """
    if isinstance(exc, EmptyVacancyError):
        return (str(exc) or "vacancy text is empty", EXIT_USER)
    if isinstance(exc, ParseError):
        return (str(exc), EXIT_USER)
    if isinstance(exc, OutputPathError):
        return (str(exc), EXIT_USER)
    if isinstance(exc, FileNotFoundError):
        name = exc.filename if exc.filename is not None else "?"
        return (f"file not found: {name}", EXIT_USER)
    if isinstance(exc, IsADirectoryError):
        name = exc.filename if exc.filename is not None else "?"
        return (f"expected a file, got a directory: {name}", EXIT_USER)
    if isinstance(exc, UnicodeDecodeError):
        return ("input is not valid UTF-8", EXIT_USER)
    if isinstance(exc, LLMError):
        return (str(exc) or "LLM request failed", EXIT_RUNTIME)
    if isinstance(exc, ConfigError):
        return (str(exc), EXIT_RUNTIME)
    if isinstance(exc, OSError):
        # BrokenPipeError is handled before _fail (quiet exit 0 for ``| head``).
        return (f"I/O error: {exc}", EXIT_USER)
    # Unknown exceptions (raw SDK/HTTP) may embed URL/headers/body with the key.
    # Show type only; details wait for --verbose in step 7.
    return (f"unexpected error ({type(exc).__name__})", EXIT_RUNTIME)


def _fail(exc: BaseException) -> NoReturn:
    """Print a single-line error to stderr and exit. No traceback."""
    message, code = map_cli_error(exc)
    # ``exc.attempts`` (ParseError / LLMError) intentionally unused here — kept
    # for step-7 logging without printing token/prompt metadata to the user.
    _write_stderr(message)
    raise typer.Exit(code)


def _exit_on_broken_pipe() -> NoReturn:
    """Quiet exit when the consumer closed stdout early (e.g. ``| head -c 10``).

    Redirect stdout to ``/dev/null`` so interpreter shutdown does not print
    ``Exception ignored … BrokenPipeError``. Exit 0: truncation was intentional.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except OSError:
        pass
    raise typer.Exit(EXIT_OK)


def _write_stderr(message: str) -> None:
    sys.stderr.buffer.write((message + "\n").encode("utf-8", errors="replace"))
    sys.stderr.buffer.flush()


def _write_stdout(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.flush()


def _read_input(source: str) -> str:
    """Read vacancy text as UTF-8 (locale-independent for both file and stdin)."""
    if source == "-":
        return sys.stdin.buffer.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def _check_output_path(output: Path | None) -> None:
    """Reject obvious ``-o`` problems before spending an LLM call."""
    if output is None:
        return
    if output.exists() and output.is_dir():
        raise OutputPathError(f"output path is a directory: {output}")
    parent = output.parent
    if parent.exists() and not parent.is_dir():
        raise OutputPathError(f"output parent is not a directory: {parent}")


def _emit_json(json_text: str, output: Path | None) -> None:
    """Write JobPosting JSON. Compact (jq-friendly); nulls kept (Decision A).

    Without ``-o``: JSON on stdout only (UTF-8 bytes).
    With ``-o``: overwrite the file (create parent dirs); stdout stays empty.
    """
    payload = json_text if json_text.endswith("\n") else json_text + "\n"
    if output is None:
        _write_stdout(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")


@app.command("parse", epilog=EXIT_CODES_EPILOG)
def parse_command(
    source: Annotated[
        str,
        typer.Argument(help="Vacancy text file path, or - to read stdin."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            help="Write JSON to this path instead of stdout (overwrites).",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Override GEMINI_MODEL for this invocation.",
        ),
    ] = None,
) -> None:
    """Parse vacancy text and emit JobPosting JSON."""
    # One catch site: every failure (including unexpected SDK bugs) goes through
    # map_cli_error via _fail — no bare traceback, fallback is live.
    try:
        text = _read_input(source)
        # Fail-fast empty check before config so exit 1 does not depend on .env.
        # Service still enforces the same invariant for non-CLI callers.
        if not text.strip():
            raise EmptyVacancyError("vacancy text is empty")
        _check_output_path(output)

        settings = load_settings()
        effective_model = model if model is not None else settings.gemini_model
        client = build_llm_client(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=effective_model,
        )
        result = parse_vacancy(text, client)
        # ``result.attempts`` reserved for step-7 logging (not printed).
        _emit_json(result.posting.model_dump_json(), output)
    except BrokenPipeError:
        _exit_on_broken_pipe()
    except Exception as exc:
        _fail(exc)


def main() -> None:
    """Entry point for ``vacancy-parser`` console script."""
    app()
