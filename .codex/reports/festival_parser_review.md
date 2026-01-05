# Universal Festival Parser - Code Review Report

**Date:** 2026-01-04  
**Branch:** `feat/universal-festival-parser`  
**Reviewer:** Automated Analysis + Manual Review

## Summary

✅ **Overall Status:** APPROVED with minor recommendations

The Universal Festival Parser implementation is well-structured with proper separation of concerns (RDR architecture), comprehensive error handling, and good test coverage. Security considerations for API key handling are properly addressed.

---

## 1. Code Quality & Best Practices

### ✅ Strengths

| Item | Status |
|------|--------|
| RDR Architecture separation | ✅ Well implemented |
| Type annotations | ✅ Comprehensive |
| Docstrings | ✅ All public functions documented |
| Error handling | ✅ Proper exception handling with logging |
| Logging | ✅ Detailed logging throughout |
| DRY principle | ✅ Good code reuse |

### ⚠️ Minor Issues

1. **`festival_parser.py:257`** - Name extraction could use walrus operator:
   ```python
   # Current
   name = festival_data.get("title_short") or festival_data.get("title_full", "")
   # Suggested (more explicit)
   if not (name := festival_data.get("title_short")):
       name = festival_data.get("title_full", "")
   ```

2. **Import placement** - Lazy imports in `process_festival_url()` could be moved to module level:
   ```python
   # Move to top of file for clarity
   from main import sync_festival_page, rebuild_fest_nav_if_changed, set_setting_value
   ```

---

## 2. Security Analysis

### ✅ API Key Handling

| Control | Implementation | Status |
|---------|----------------|--------|
| Two-dataset split | Cipher + Key in separate datasets | ✅ |
| In-memory only | Never written to disk | ✅ |
| Fernet encryption | Industry standard | ✅ |
| Private datasets | Recommended in docs | ✅ |
| Fallback chain | Env → Kaggle Secrets → Datasets | ✅ |

### ✅ No Secrets in Logs

- LLM logs store prompts/responses but NOT API keys
- `llm_log.json` is safe to share with operators

### ⚠️ Recommendation

Add explicit sanitization for error messages:
```python
# In reason.py, before logging errors
error = f"Gemma API error: {e}"
# Ensure API key is not included in exception message
if "AIza" in str(e):
    error = "Gemma API error: [REDACTED]"
```

---

## 3. Error Handling Completeness

### ✅ Covered Scenarios

| Scenario | Handler |
|----------|---------|
| Network failure | `render_page()` try/catch |
| Invalid HTML | `distill_html()` fallback |
| LLM JSON parse error | `reason_with_gemma()` returns error tuple |
| Rate limit exceeded | `GemmaRateLimiter` with wait |
| Kaggle job failure | `RuntimeError` with run_id |
| Missing UDS output | `RuntimeError` with run_id |
| Supabase upload failure | Logged, continues with None URL |

### ⚠️ Suggestions

1. Add timeout for individual LLM calls in `reason.py`
2. Consider retry logic for transient Supabase failures

---

## 4. Type Annotations

### ✅ Coverage: 95%+

All public functions are annotated. Key signatures:

```python
def classify_source_type(url: str) -> Literal["canonical", "official", "external"]: ...
async def process_festival_url(...) -> tuple["Festival", str | None, str | None]: ...
async def save_to_supabase_storage(...) -> tuple[str | None, str | None]: ...
```

### ⚠️ Minor Gap

Add `Protocol` type for callback in `process_festival_url`:
```python
from typing import Protocol

class StatusCallback(Protocol):
    async def __call__(self, status: str) -> None: ...
```

---

## 5. Test Coverage

### Current Coverage

| Module | Unit Tests | Integration | E2E |
|--------|------------|-------------|-----|
| `festival_parser.py` | ✅ 16 tests | ⏳ | ⏳ |
| `date_utils.py` | ✅ 13 tests | N/A | N/A |
| Kaggle notebook | ⏳ | ⏳ | ⏳ |
| Bot handlers | ⏳ | ⏳ | ✅ Gherkin |

### ⚠️ Recommended Additional Tests

1. **`test_upsert_festival_from_uds`** - Mock database to test create vs update
2. **`test_save_to_supabase_storage`** - Mock Supabase client
3. **`test_process_festival_url_failure`** - Test Kaggle failure path

---

## 6. Recommendations Summary

| Priority | Item | Difficulty |
|----------|------|------------|
| LOW | API key sanitization in error logs | Easy |
| LOW | LLM call timeout | Easy |
| MEDIUM | Additional unit tests for upsert/storage | Medium |
| LOW | Import organization | Easy |

---

## 7. Files Changed

```
source_parsing/
├── festival_parser.py    # Main pipeline (560 lines)
├── date_utils.py         # Date formatting (109 lines)
├── kaggle_runner.py      # +dataset_sources param

kaggle/UniversalFestivalParser/
├── 9 Python modules (1457 lines total)

main_part2.py
├── URL detection + handler
├── Edit menu extensions

tests/
├── test_festival_parser.py (29 tests)
├── test_festival_date_format.py
├── e2e/features/festival_parser.feature

docs/
├── FESTIVAL_PARSER.md
```

---

## Conclusion

The implementation is production-ready with the caveats noted. The RDR architecture provides clear separation, LLM logging enables debugging, and security measures are appropriate for the use case.

**Approval:** ✅ Ready for merge after addressing high/medium priority items.
