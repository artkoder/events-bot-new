# Security Review Request: Supertester E2E Role

## Context

We're adding a "supertester" role for E2E testing that grants admin access in development mode only.

## Implementation

### New functions in `main.py`:

```python
def is_e2e_tester(user_id: int) -> bool:
    """Check if user is the E2E tester (only works in DEV_MODE)."""
    if os.getenv("DEV_MODE") != "1":
        return False
    tester_id = os.getenv("E2E_TESTER_ID")
    if not tester_id:
        return False
    try:
        return int(tester_id) == user_id
    except ValueError:
        return False


def has_admin_access(user) -> bool:
    """Check if user has admin access (superadmin or E2E tester in DEV_MODE)."""
    if user is None:
        return False
    if user.is_superadmin:
        return True
    return is_e2e_tester(user.user_id)
```

## Review Questions

1. **Environment variable security**: Is checking `DEV_MODE != "1"` sufficient? Could a malicious actor set this in production?

2. **Race conditions**: Could there be timing issues with environment variable reads?

3. **Audit trail**: Should we log when E2E tester accesses admin functions?

4. **Scope limitation**: Should E2E tester have ALL admin permissions or a subset?

5. **Edge cases**: What if `E2E_TESTER_ID` contains spaces, invalid chars, or multiple IDs?

6. **Command coverage**: Which commands need to use `has_admin_access()` vs keeping `is_superadmin` check?

## Files to review

- `/workspaces/events-bot-new/main.py` - lines 451-483 (new functions)
- All places where `is_superadmin` is checked

## Expected output

Please analyze security implications and make any necessary code changes. Provide a report of:
1. Security issues found
2. Changes made
3. Recommendations for additional hardening
