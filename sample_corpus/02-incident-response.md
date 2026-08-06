# Incident Response Runbook

## Severity definitions

A SEV-1 is a complete production outage, confirmed data loss, or an active security breach. A SEV-2 is a major degradation affecting more than 20 percent of customers without a reasonable workaround. A SEV-3 is a limited degradation or defect with a practical workaround.

## Declaring an incident

Declare incidents in the `#incidents` Slack channel using `/incident declare`. The incident commander opens a video bridge, assigns an operations lead and communications lead, and starts the incident document. For SEV-1 incidents, notify the executive on-call within ten minutes.

## Communication cadence

Customer-facing updates are published every 30 minutes for SEV-1 incidents and every 60 minutes for SEV-2 incidents. Updates must state current impact, mitigation progress, and the next update time. Never speculate about root cause before evidence is confirmed.

## Resolution and follow-up

An incident is resolved only after customer impact has ended and monitoring has remained healthy for 30 minutes. SEV-1 post-incident reviews are due within three business days. SEV-2 reviews are due within five business days. Action items require an owner and target date.

## Common telemetry failures

Alert code `OBS-1042` means the metrics gateway rejected an expired service token. Rotate the `METRICS_PUSH_TOKEN`, restart the affected exporter, and confirm two consecutive successful pushes. Alert code `DB-7721` means replica lag exceeded 45 seconds; pause background migrations before considering a primary failover.

