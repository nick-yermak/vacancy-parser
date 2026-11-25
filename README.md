# vacancy-parser

Parse job postings with Gemini into a structured `JobPosting` schema.

## CLI

```bash
uv run vacancy-parser parse <file|-> [-o out.json] [--model MODEL]
```

Exit codes (also shown in `--help` epilog):

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Input/validation: empty text, missing/non-UTF-8 file, bad `-o`, `ParseError`. Ctrl+C also exits 1 (`Aborted!` from Click). |
| 2 | Config, LLM/API, or unexpected internal/SDK errors. Click/Typer usage errors (bad argv) also exit 2 before our handler — usage mistakes are deterministic, so we keep Click’s code rather than remapping. |

Stdout is JSON only (pipe-friendly for `jq`). Human messages go to stderr. No tracebacks; the API key never appears in errors, `--help`, or `repr`.

## Design decisions

- **F (validation):** The LLM client returns raw text + call metadata. The service
  layer validates into a Pydantic model and owns retry. The fake can therefore
  script invalid JSON for retry tests.
- **G (schema typing):** `LLMClient.complete` takes `type[BaseModel]` without a
  TypeVar — the client always returns `LLMResponse`. The service also keeps a
  concrete `JobPosting` in `ParseResult`; a generic pays only if a caller-chosen
  schema drove the return type.
- **H (fake location):** `FakeLLMClient` lives in `src/vacancy_parser/llm.py` so
  service tests (and optional dry-run) can use it without importing from `tests/`.
- **L (API surface):** Phase 0 uses `generateContent` with
  `response_mime_type` + `response_json_schema`. Interactions API is better for
  agentic/multi-turn work later; generateContent stays fully supported in
  google-genai 3.6.1 and matches our single-turn extract.
- **M (usage tokens):** `input_tokens` / `output_tokens` are `int | None`. Missing
  usage must not become `0` — step-7 cost logs would understate spend. Empty /
  MAX_TOKENS replies still bill; usage is carried on `LLMError.partial`.
- **N (errors):** One `LLMError` with `status` and `retryable`. `attempts` is
  service metadata attached onto the same exception so subclasses/fields survive.
- **O (transport retries):** Own loop with injectable sleep; honor Gemini
  `retryDelay`, else 15s (429/RPM) / 2s (5xx/network). Skip PerDay 429 and
  400/401/403. `transport_calls` exposes HTTP tries (incl. retries) for step-7
  RPD. SDK `HttpRetryOptions` stay unset. CLI must not print `__cause__`
  tracebacks (may contain the API key). Non-transport SDK exceptions still
  propagate out of `llm.py`; the CLI catches them once and maps to exit 2.
- **P (CLI test seam):** `build_llm_client()` is the production factory; tests
  monkeypatch it to return `FakeLLMClient` (Angular `TestBed.overrideProvider`).
  The CLI itself never imports fakes or grows an `if testing` branch.
- **Q (JSON / `-o`):** `model_dump_json()` with defaults — compact, nulls kept
  (Decision A). Existing `-o` files are overwritten; parent dirs are created.
  With `-o`, stdout stays empty; without it, JSON goes to stdout for `| jq`.
  Obvious `-o` failures (path is a directory) are checked before the LLM call
  so quota is not spent on a doomed write.
- **R (exit code 2 collision):** Click/Typer usage errors exit 2 before our
  callback (`ClickException` is public API). We leave that code as-is because a
  usage mistake is deterministic from argv — remapping adds wrapper complexity
  for little gain. Our mapped config/LLM/unexpected failures also use 2; exit 1
  stays reserved for vacancy input/validation. Documented here and in `--help`
  epilog.
