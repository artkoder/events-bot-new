# Cron Schedule

The bot uses APScheduler to run periodic maintenance tasks every 15 minutes. Each job checks whether its specific conditions are met and exits quickly if not.

## Jobs

- **partner reminders** – reminds inactive partners after 09:00 local time.
- **cleanup old events** – removes past events after 03:00 local time and notifies the superadmin.
- **VK daily posts and polls** – publishes daily announcements and festival polls when posting times are reached and a VK group is configured.
- **Telegraph pages sync** – refreshes month and weekend Telegraph pages after 01:00 local time. Disabled by default; enable with `ENABLE_NIGHTLY_PAGE_SYNC=1`. Nightly runs update both page content and the month navigation block.
- **festival navigation rebuild** – rebuilds festival navigation and landing page nightly.

## Environment variables

- `VK_USER_TOKEN` – user token for VK posts (scopes: wall,groups,offline).
- `VK_TOKEN` – optional group token used as a fallback.
- `EVBOT_DEBUG` – enables extra logging and queue statistics.

All jobs are lightweight and finish in under a few seconds so that the bot remains responsive.
