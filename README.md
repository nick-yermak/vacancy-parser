# vacancy-parser

Parse job postings with Gemini into a structured `JobPosting` schema.

## Design decisions

- **F (validation):** The LLM client returns raw text + call metadata. The service
  layer validates into a Pydantic model and owns retry. The fake can therefore
  script invalid JSON for retry tests.
- **G (schema typing):** `LLMClient.complete` takes `type[BaseModel]` without a
  TypeVar — the return type does not depend on the schema. A generic `T` belongs
  on a service that returns a validated model.
- **H (fake location):** `FakeLLMClient` lives in `src/vacancy_parser/llm.py` so
  service tests (and optional dry-run) can use it without importing from `tests/`.
