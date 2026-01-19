# Recurring Events

This document describes the proposed implementation for supporting events that repeat on a regular schedule. It is written for a separate development team and may evolve as the feature is refined.

## Overview

Some announcements represent a series of identical events occurring on specific days of the week. These "recurring" events should be tracked without crowding the main `/events` listing. Moderators must confirm that each occurrence actually takes place and can pause the series if needed.

## Requirements

- **Duration limit** – A recurring event is created for no more than three months at a time. The end date can be adjusted manually.
- **Start date** – Events may begin in the future (e.g. starting from 1 June). A start date is stored along with the recurrence pattern.
- **Exclusions** – Moderators may skip individual dates or pause the series for a period.
- **Moderator command** – A new command lists all active recurring events with buttons:
  - **Confirm** – acknowledges that the next occurrence is happening as scheduled.
  - **Pause / Resume** – temporarily disable or re-enable the series.
- **Monthly reminder** – Once a month the bot sends a message to all moderators reminding them to review recurring events and instructs how to invoke the command above.
- **Not in `/events`** – Recurring entries do not appear in the regular `/events` command output.

## Data Model

A new table stores recurring events with the following fields:

- `title`, `short_description`, `location_name`, `city` – same as regular events.
- `weekday_mask` – which days of the week the event occurs (e.g. Monday and Thursday).
- `time` – start time or range.
- `start_date` – optional date when the series begins.
- `last_date` – automatically set to three months from creation and editable by moderators.
- `is_paused` – whether the series is currently suspended.

Individual occurrences are not stored as separate events unless the moderator confirms them.

## Workflow

1. Moderator creates a recurring event using a yet-to-be-defined command or interface.
2. The bot schedules reminders to check upcoming occurrences.
3. Moderators periodically run the management command, confirm ongoing events, and adjust dates if necessary.
4. When a series ends or is cancelled permanently, it can be removed.

The exact command names and button layouts are left for implementation but should follow existing conventions.
