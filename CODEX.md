# CODEX

## Mandatory Rules
- **ALWAYS** review this CODEX before starting any task on this repository.
- **ALWAYS** keep documentation and automation artifacts synchronized with code changes.
- **ALWAYS** run and update relevant smoke tests before requesting review or completing work.
- **NEVER** merge or submit changes without passing smoke tests and linting checks.
- **NEVER** introduce fixtures or test data that persist outside their intended scope.

## Definition of Done
Every change must satisfy **all** of the following before it is considered complete:
1. Smoke tests are executed and passing, or a justified exception is documented in the change description.
2. Test fixtures are cleaned up, scoped appropriately, and free of side effects across the suite.
3. The README, user-facing help, and CHANGELOG entries are updated (or explicitly confirmed as not needed) to reflect behavioral or interface changes.

## Critical Code Paths
The following modules and files are considered critical paths and demand extra scrutiny, regression testing, and reviewer visibility whenever touched:
- `main`
- `db`
- `imagekit_poster`
- `vk_intake`
- `vk_review`
- `markup`
- `scheduling`
- `digests`
- `sections`
- `shortlinks`
- `supabase_export`
- `safe_bot`
- `span`
- `net`
- `models`

## Agent Guidance
Future contributors and agents must treat this CODEX as required reading prior to any repository work. Confirm in your task notes that you have reviewed it.
