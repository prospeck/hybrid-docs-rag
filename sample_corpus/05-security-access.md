# Security and Access Standard

## Production access

Production access is role-based, time-limited, and approved through AccessHub. Standard production access expires after eight hours. Emergency break-glass access expires after one hour and automatically opens a review ticket. All production sessions are recorded by the Session Broker.

## Approval requirements

Read-only production access requires approval from the requester's manager and the service owner. Write access additionally requires the current incident commander or Security on-call. Requests must include a ticket, business justification, target service, and requested duration.

## Secrets

Secrets must never appear in source code, chat, tickets, or application logs. Rotate production secrets every 90 days and immediately after suspected disclosure. Local development uses short-lived developer credentials issued by AccessHub; copying production secrets into local environment files is prohibited.

## Data classification

Data is classified as Public, Internal, Confidential, or Restricted. Customer authentication data, payment data, private keys, and production credentials are Restricted. Restricted data must be encrypted in transit and at rest, and access logs are retained for 400 days.

## Vulnerability remediation

Critical vulnerabilities with known exploitation must be mitigated within 24 hours. Other critical vulnerabilities are due within seven days, high severity within 30 days, medium within 90 days, and low within 180 days.

