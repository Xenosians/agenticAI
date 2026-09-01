# Security Policy

## Project Status

ITSM Agent is currently a local development prototype.

The project now includes experimental multi-agent hub/worker functionality in addition to the original single-agent runtime.

Do not deploy the current repository directly into a production Active Directory environment.

## Core Security Principle

Language models are not authorization systems.

The project follows this rule:

```text
Models propose.
Application code validates and authorizes.
MCP executes.
Directory permissions enforce.
```

Neither the hub model nor any specialist worker may directly grant itself additional privileges.

## Multi-Agent Trust Model

The Hub is responsible for selecting specialist workers.

Specialist workers are responsible for proposing narrow operations.

Neither layer is trusted to authorize privileged execution.

A worker proposal must pass through application-controlled validation before MCP execution.

```text
Hub
 |
 v
Worker
 |
 v
Structured proposal
 |
 v
ToolGateway
 |
 +-- agent allowlist
 +-- tool registry
 +-- identifier validation
 +-- approval policy
 |
 v
MCP
```

## Agent Tool Isolation

Each specialist has an explicit tool allowlist.

For example, an Account specialist may be configured with:

```text
account_status
unlock_user
reset_password
```

Application code verifies the requested tool against this list.

A prompt instruction such as:

```text
Only use account tools.
```

is not considered sufficient enforcement.

The `ToolGateway` must reject any tool outside the registered specialist allowlist.

## Hub Routing Validation

Hub output is treated as untrusted structured data.

When the model proposes specialist names:

```json
{
  "agents": [
    "account-specialist",
    "unknown-agent"
  ]
}
```

only names already present in the `AgentRegistry` may be accepted.

Models may not create new runtime privileges by inventing specialist names.

## Structured Worker Output

Worker function calls must pass structured-output validation before execution.

Malformed JSON, missing tool names, invalid argument objects, or unsupported multi-call output must not reach MCP.

The current worker runtime intentionally allows only one tool call per `AgentTask`.

## Identifier Validation

Identity-sensitive arguments proposed by a model must correspond to identifiers present in the original user request.

Example:

```text
User:
Is jdoe locked?

Worker proposes:
account_status(user_id="jsmith")
```

The operation must be rejected.

This protects against silent model substitution of account targets.

## Human Approval Boundary

Mutation tools remain protected by application-side approval checks.

Current mutation tools include:

* `unlock_user`
* `reset_password`

A worker proposal does not constitute approval.

Expected mutation flow:

```text
Worker proposes mutation
        |
        v
ToolGateway
        |
        v
Pending approval
        |
        v
Human approval
        |
        v
MCP execution
        |
        v
Directory result verification
```

The language model must not be able to bypass this process by changing its output format or wording.

`reset_password` should remain disabled for real LDAP execution until password policy, credential handling, delegated permissions, and post-change verification are implemented and tested.

## MCP Boundary

Specialists must not receive unrestricted direct access to privileged directory backends.

Permitted operations cross the MCP boundary after application-side validation.

The MCP layer remains separate from model inference and orchestration.

## Directory Accounts

Use separate identities for:

* read-only directory queries
* approved directory mutations

Write identities should receive only the permissions required by supported mutation tools.

Do not use Domain Administrator credentials in production.

Development-only Administrator access used with the local Samba environment must not be copied into a production deployment.

## Secrets

Never commit:

* `.env`
* LDAP bind passwords
* Samba administrator passwords
* production Active Directory credentials
* model/API credentials
* private keys
* generated certificates containing private keys
* authentication tokens

Use `.env.example` only for variable names and safe placeholder values.

If a credential is accidentally exposed publicly, rotate it immediately.

Removing a secret from the latest Git commit does not guarantee that it has been removed from repository history.

## Local Models

Local model checkpoints should not contain credentials or sensitive directory data.

Do not fine-tune or persist prompts containing production credentials, passwords, authentication tokens, or unnecessary personal information.

Model output should be treated as untrusted application input.

## LDAP Transport

Use encrypted LDAP transport.

The local Samba development environment exposes LDAPS.

Development self-signed certificates may require special handling, but production certificate validation must not be disabled.

## Docker

The local Samba AD container uses development-oriented configuration and may use elevated container privileges to support provisioning.

Treat the environment as disposable development infrastructure.

Do not reuse the Compose configuration unchanged in production.

## Logging

Avoid logging:

* passwords
* raw authentication headers
* private keys
* complete credential-bearing environment variables
* password-reset values
* model prompts containing sensitive credentials

Worker and Hub debugging logs should also avoid unnecessary sensitive information.

Structured audit records for privileged actions should eventually contain:

* requested action
* target
* requesting identity
* selected specialist
* proposed tool
* approval record
* approver
* execution time
* execution result

without storing secret password material.

## Approval and Audit Persistence

Current approval handling is development-oriented and is not yet a durable production audit system.

Before production use, approval and execution records should be stored in an append-only or appropriately protected persistent audit store.

## Multi-Agent Expansion

Adding a new specialist should require:

1. registration in the Agent Registry
2. an explicit model assignment
3. an explicit tool allowlist
4. structured worker output
5. ToolGateway enforcement
6. unit tests
7. integration tests before privileged execution is enabled

A newly added model must not automatically inherit all available MCP tools.

## API / Microservice Security

A future REST or microservice layer may be implemented separately from the Python agent runtime.

If an Elixir/Phoenix or other API service is introduced, it should add—not replace—security controls such as:

* authenticated callers
* authorization
* rate limiting
* request correlation
* secure service-to-service authentication
* TLS
* audit context
* request size limits

The external API must not expose raw ToolGateway or MCP execution directly to untrusted clients.

## Production Requirements

Before production deployment, the project still requires at minimum:

* production authentication
* authorization
* durable audit persistence
* durable approval persistence
* least-privilege write-service accounts
* production TLS validation
* secret management
* nested-group/access-policy review
* mutation verification
* password-reset security design
* model-output robustness testing
* Hub routing validation
* specialist-worker validation
* API/service security controls

## Reporting Security Problems

Do not publish credentials, private keys, production directory information, or exploitable configuration in public issues.

Security reports should contain the minimum reproduction information necessary and redact all secrets.
