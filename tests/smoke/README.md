# Smoke test suite

This directory contains a small collection of "confidence" checks that run very
quickly and **must succeed offline**.  They are intended to catch obvious
regressions before we invest in heavier integration or network-dependent
pipelines.

## Current coverage

* `test_module_imports.py` – parametrically imports the most critical runtime
  entry points (bot command modules, scheduling helpers, exporters, etc.) to
  verify they can be loaded without executing side effects.

Because these tests import production code directly, they double as a guard
against accidentally adding import-time network calls, environment detection, or
other behaviour that would break isolated execution.

## Adding new smoke scenarios

When expanding this suite, keep the following principles in mind:

1. **Offline execution first.**  Smoke tests run in environments without network
   access or external credentials.  If a code path normally performs I/O,
   either refactor the production code to defer that work or use existing
   fixtures/mocking utilities from `tests/_helpers` to stub the dependency.
2. **Prefer shared fixtures.**  If a scenario requires objects such as
   pre-populated databases, feature flags, or frozen time, add or reuse fixtures
   in `tests/conftest.py` or the `_helpers` package instead of hand-rolling
   setup in each test.  This keeps smoke tests declarative and ensures they
   compose with the broader pytest ecosystem.
3. **Focus on expectations, not exhaustiveness.**  Smoke scenarios should target
   the minimal behaviour necessary to confirm a feature still "turns on"—for
   example: "command module registers handlers", "scheduler builds a job list",
   or "renderer returns non-empty markup".  Deeper behavioural checks belong in
   the functional and integration suites under `tests/`.
4. **Document future intent.**  When a smoke test highlights a missing scenario,
   leave a short TODO describing the desired coverage (e.g. handling new
   message types or onboarding flows).  This keeps the suite aligned with the
   product roadmap and makes it clear which gaps are deliberate.

Following these guidelines will keep the smoke suite fast, deterministic, and a
useful early-warning layer as we add new functionality.
