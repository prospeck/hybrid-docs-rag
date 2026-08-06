# Platform Deployment Guide

## Environments

The supported environments are development, staging, and production. Staging mirrors production topology but uses anonymized data. Direct developer access to production containers is prohibited; all operational access must use the audited Session Broker.

## Release sequence

Create an immutable image, run unit and integration tests, scan the image, deploy to staging, and complete the smoke suite. Production uses a 10 percent canary for 15 minutes, followed by 50 percent for 15 minutes, then 100 percent. Error rate must remain below 1 percent and p95 latency must remain below 400 milliseconds at each gate.

## Rollback

Set `DEPLOYMENT_HALT=true` to stop further rollout steps. Roll back when error rate exceeds 1 percent for five consecutive minutes, p95 latency exceeds 400 milliseconds for ten minutes, or a new SEV-1 is linked to the release. Database migrations must be backward compatible for at least one release.

## Feature flags

Every risky customer-visible feature must use a server-side feature flag. Flags need an owner, expiry date, and removal ticket. Emergency disablement uses the configuration key `FEATURE_KILL_SWITCH`; changes propagate globally within 90 seconds.

## Build provenance

Production images must be signed by the CI workload identity. The admission controller rejects unsigned images with policy code `SIG-REQUIRED-403`. Rebuilding the same Git commit does not guarantee the same digest, so release tickets record both commit SHA and image digest.

