# Documentation Reorganization Report (2026-01-07)

## Overview
The documentation structure has been reorganized to improve navigability and separate implemented features from backlog items.

## New Structure
- **`docs/architecture/`**: High-level system overview (`overview.md`).
- **`docs/operations/`**: Guides for running and maintaining the bot (`commands.md`, `cron.md`, `e2e-testing.md`, `prod-data.md`).
- **`docs/llm/`**: LLM-related documentation (`prompts.md`, `request-guide.md`, `topics.md`).
- **`docs/reference/`**: Static reference data (`locations.md`, `holidays.md`, etc.).
- **`docs/pipelines/`**: Documentation for various parsing pipelines (`source-parsing.md`, `festival-parser.md`, etc.).
- **`docs/features/`**: Feature documentation that reflects current behavior (`digests/`, `tourist-label/`).
- **`docs/backlog/linear/`**: Detailed Linear task descriptions for upcoming work (`EVE-11`, `EVE-54`, `EVE-55`).
- **`docs/reports/`**: Historic reports and plans.

## Backlog Items
The following Linear tasks have been extracted into individual files in `docs/backlog/linear/`:
- [EVE-11: Global LLM Rate Limits](docs/backlog/linear/EVE-11-llm-rate-limits.md) (Not Implemented)
- [EVE-54: Kaggle Secrets Framework](docs/backlog/linear/EVE-54-kaggle-secrets.md) (Not Implemented)
- [EVE-55: Gemma Integration](docs/backlog/linear/EVE-55-gemma-integration.md) (Not Implemented)

## Navigation
A new section "How to Navigate Documentation" has been added to `README.md` to guide users through this new structure.
