Aligned admin gating to use `has_admin_access` so the DEV_MODE E2E tester can exercise admin flows, and updated the “skip superadmin notification” logic to treat the E2E tester like an admin. Wrote the full security review report with findings and recommendations.

**Changes**
- `main.py`: admin-only handlers and menus now call `has_admin_access`; `notify_event_added` skips notifications for E2E admin usage.
- `.codex/reports/supertester_security.md`: full security review report.

**Tests**
- Not run (not requested).

Next steps if you want:
1. Review `.codex/reports/supertester_security.md` and decide on the hardening recommendations (prod guardrail, audit logging).
2. Run a quick DEV_MODE admin/E2E smoke test to confirm the expected access paths.