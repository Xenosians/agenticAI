# Security Policy

## Project Status

ITSM Agent is currently a development prototype.

Do not deploy the current repository directly into a production Active Directory environment.

## Secrets

Never commit:

- `.env`
- LDAP bind passwords
- Samba administrator passwords
- production Active Directory credentials
- generated private keys or certificates
- model/API credentials

Use `.env.example` only for variable names and safe placeholder values.

If a credential is accidentally committed or pasted into a public location, rotate it immediately. Removing it from the latest commit is not sufficient because it may remain in Git history.

## Directory Accounts

Use separate identities for:

- read-only directory queries
- approved directory mutations

The write identity should be delegated only the permissions required by supported mutation tools.

Do not use Domain Administrator credentials in production.

## Human Approval Boundary

Mutation tools must remain protected by application-side approval checks.

The language model must not be able to bypass the approval mechanism by changing its output.

Current mutation tools include:

- `unlock_user`
- `reset_password`

`reset_password` should remain disabled in the real LDAP backend until password policy, secret handling, and delegated permissions are implemented and tested.

## LDAP Transport

Use encrypted LDAP transport.

The local Samba environment exposes LDAPS for development. Certificate verification may require special handling with a local self-signed development certificate.

Do not disable certificate verification in production.

## Docker

The local Samba AD development container currently uses elevated container privileges to support AD provisioning.

Treat this as development-only.

Do not reuse that Compose configuration unchanged in production.

## Logging

Avoid logging:

- passwords
- raw authentication headers
- private keys
- complete credential-bearing environment variables

Audit records for mutations should contain enough information to determine:

- requested action
- target
- requester
- approver
- execution time
- result

without recording sensitive password material.

## Reporting Security Problems

Do not publish credentials or exploitable production configuration in a public issue.

When reporting a security problem, include the minimum reproduction details necessary and redact all secrets.
