# Universal Festival Parser Code Review Report

## Scope
- kaggle/UniversalFestivalParser/src/config.py
- kaggle/UniversalFestivalParser/src/distill.py
- kaggle/UniversalFestivalParser/src/llm_logger.py
- kaggle/UniversalFestivalParser/src/rate_limit.py
- kaggle/UniversalFestivalParser/src/reason.py
- kaggle/UniversalFestivalParser/src/render.py
- kaggle/UniversalFestivalParser/src/secrets.py
- kaggle/UniversalFestivalParser/src/uds.py
- kaggle/UniversalFestivalParser/universal_festival_parser.py
- source_parsing/festival_parser.py

## Findings (ordered by severity)
### High
- kaggle/UniversalFestivalParser/src/rate_limit.py: GemmaRateLimiter.acquire was async but returned a context manager. The call sites used `async with rate_limiter.acquire(...)`, which would raise at runtime because the coroutine does not implement `__aenter__`.

### Medium
- kaggle/UniversalFestivalParser/src/llm_logger.py and kaggle/UniversalFestivalParser/src/reason.py: manual `__enter__`/`__exit__` calls suppressed exceptions, so failed LLM calls were logged as success. Response token overrides were also ignored, making usage stats inaccurate.
- kaggle/UniversalFestivalParser/src/render.py: HTTP 4xx/5xx responses set `error` but still marked `success=True`, so the pipeline could proceed on failed renders.
- kaggle/UniversalFestivalParser/src/reason.py: validation pass only stripped ```json fences, not plain ``` fences, which can cause JSON parsing failures.

### Low
- Typing is loose in several places (e.g., `llm_logger` parameters are untyped, and generic `dict`/`Any` is used in UDS link fields). This is not a bug but reduces static checking value.
- Tests for render/distill/reason are missing due to external dependencies (Playwright, BeautifulSoup, and google.generativeai). Coverage is currently focused on integration and orchestration, not these Kaggle-only modules.

## Fixes Applied
- Made `GemmaRateLimiter.acquire` synchronous so `async with` works correctly.
- Updated LLM tracking to use context managers so exceptions are captured; response token overrides now take precedence over estimates.
- Marked render success based on whether an HTTP error was detected; added warning logging on HTTP error.
- Normalized code fence stripping in the validation pass.

## Tests Added
- Added `tests/test_universal_festival_parser_utils.py` to cover:
  - LLMLogger response token overrides
  - LLMLogger error capture on exceptions
  - GemmaRateLimiter context manager usage
  - get_api_key env override

## Tests Run
- `pytest tests/test_universal_festival_parser_utils.py`

## Security Review
- No hard-coded secrets were found in `secrets.py`. API keys are sourced from env vars, Kaggle secrets, or private datasets.
- LLM logs store full prompts and responses; if prompts can contain sensitive data, ensure storage permissions and retention are appropriate.

## Recommended Follow-ups
- Add mock-based tests for `render.py`, `distill.py`, and `reason.py` to exercise error paths without external services.
- Consider adding optional redaction or size caps for LLM logs if storage or privacy becomes a concern.
