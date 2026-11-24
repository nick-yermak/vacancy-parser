# vacancy-parser

Parse job postings with Gemini into a structured `JobPosting` schema.

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
  tracebacks (may contain the API key).
