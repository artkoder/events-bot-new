# VK metrics

The application exposes a text endpoint `/metrics` suitable for scraping.

## Metrics

- `vk_fallback_group_to_user_total{method="<name>"}` – number of times a VK API
  call had to fall back from a group token to a user token for a given method.
- `vk_intake_processing_time_seconds_total` – cumulative time spent converting
  VK posts into events.
