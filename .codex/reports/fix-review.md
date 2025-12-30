**Findings**
- High — Pending jobs now keep a future `next_run_at` and ignore earlier values, so callers cannot force an immediate run when a deferred pending job already exists (e.g., manual rebuilds via `ics_fix_nav`). This is a behavior change vs. “reset to now”. `main.py:9978` `main.py:13097`
- Medium — Follow-up jobs rely on `depends_on` for ordering, but the worker doesn’t enforce dependencies; a follow-up can run immediately and then be overwritten by a still-running owner. This risk now extends to week/weekend pages. `main.py:10018` `main.py:11606`
- Low — `set_setting_value` catches all `Exception` types and retries inserts; unexpected DB errors can be masked and add extra round-trips. `main.py:2383`
- Low — `MONTH_KEY_PATTERN` validates only shape, not ranges, so invalid dates like `2025-13` still pass. `main.py:2428`

**Open Questions / Assumptions**
- Is preserving deferral intended to block “force now” behavior even for manual rebuild tools, or should callers be able to override pending deferred jobs?
- Are `depends_on` semantics intentionally advisory-only, or should they be enforced for nav follow-ups?

**Change Summary**
- Review report saved to `.codex/reports/fix-review.md`.
- Tests not run (review only).

Next steps (pick any):
1) Decide whether to allow “force now” for pending deferred jobs and adjust `enqueue_job` accordingly.  
2) If follow-ups must run after owners, add dependency enforcement in the worker or equivalent gating.  
3) Run the deferred rebuild tests to validate behavior in your environment.