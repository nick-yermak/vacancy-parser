"""CLI composition root: read input → settings → LLM client → parse → JSON.

No business logic here — only wiring, I/O, and exit-code mapping.
"""

from __future__ import annotations

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

app = typer.Typer(
    name="vacancy-parser",
    help="Parse job vacancy text into structured JobPosting JSON via Gemini.",
    no_args_is_help=True,
    # Locals can hold the API key; never pretty-print them. Disable rich traces
    # entirely — we map known errors ourselves and must not leak a traceback.
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _root() -> None:
    """Parse job vacancy text into structured JobPosting JSON via Gemini.

    A no-op callback keeps ``parse`` as a real subcommand. Without it, Typer
    collapses a single ``@app.command`` into the root (``vacancy-parser <file>``),
    which would break the documented ``vacancy-parser parse …`` entry.
    """


def build_llm_client(*, api_key: str, model: str) -> LLMClient:
    """Construct the production Gemini client.

    Tests monkeypatch this factory (Angular-style provider override) so the CLI
    module never grows a ``if testing`` branch.
    """
    return GeminiLLMClient(api_key=api_key, model=model)


def map_cli_error(exc: BaseException) -> tuple[str, int]:
    """Map an exception to ``(stderr message, exit code)`` — single source of truth.

    Exit 1 — caller/input/validation problems.
    Exit 2 — config or LLM/API failures.

    ``ParseError.attempts`` / ``LLMError.attempts`` stay on ``exc`` for step-7
    logging; this function only chooses the human message and code.
    """
    if isinstance(exc, EmptyVacancyError):
        return (str(exc) or "vacancy text is empty", EXIT_USER)
    if isinstance(exc, ParseError):
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
        return (f"I/O error: {exc}", EXIT_USER)
    return (str(exc) or type(exc).__name__, EXIT_USER)


def _fail(exc: BaseException) -> NoReturn:
    """Print a single-line error to stderr and exit. No traceback."""
    message, code = map_cli_error(exc)
    # ``exc.attempts`` (ParseError / LLMError) intentionally unused here — kept
    # for step-7 logging without printing token/prompt metadata to the user.
    typer.echo(message, err=True)
    raise typer.Exit(code)


def _read_input(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    return Path(source).read_text(encoding="utf-8")


def _emit_json(json_text: str, output: Path | None) -> None:
    """Write JobPosting JSON. Compact (jq-friendly); nulls kept (Decision A).

    Without ``-o``: JSON on stdout only.
    With ``-o``: write/overwrite the file (create parent dirs); stdout stays empty
    so exit code is the success signal.
    """
    payload = json_text if json_text.endswith("\n") else json_text + "\n"
    if output is None:
        sys.stdout.write(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")


@app.command("parse")
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
    try:
        text = _read_input(source)
    except (OSError, UnicodeDecodeError) as exc:
        _fail(exc)

    try:
        settings = load_settings()
    except ConfigError as exc:
        _fail(exc)

    effective_model = model if model is not None else settings.gemini_model
    client = build_llm_client(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=effective_model,
    )

    try:
        result = parse_vacancy(text, client)
    except (EmptyVacancyError, ParseError, LLMError) as exc:
        _fail(exc)

    # ``result.attempts`` reserved for step-7 logging (not printed).
    json_text = result.posting.model_dump_json()
    try:
        _emit_json(json_text, output)
    except OSError as exc:
        _fail(exc)


def main() -> None:
    """Entry point for ``vacancy-parser`` console script."""
    # Typer/Click usage errors still exit 2 (Click convention). Our mapped
    # ConfigError/LLMError also use 2 — see review note R.
    app()
