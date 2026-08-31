# ITSM Agent

A local-first IT service management agent that uses a small local language model to interpret natural-language requests and execute directory-service tools through MCP.

The project is currently a development prototype focused on safe, auditable Active Directory-style workflows using a local Samba AD environment.

## Current Status

Working pieces:

- Local Qwen3-0.6B inference
- Tool selection with a small `ACTION` / `ARGS` / `FINAL` protocol
- Persistent MCP subprocess for tool execution
- Read vs. mutation risk separation
- Human approval gate for mutation tools
- Identifier guardrails before tool execution
- `DirectoryService` abstraction
- Mock directory backend
- LDAP/Samba AD backend
- Local Samba Active Directory domain in Docker
- LDAPS connectivity and read-only LDAP preflight
- Account status lookup
- Group-based access checks
- Experimental approved account-unlock mutation

Not enabled yet:

- Real password reset
- Production Active Directory integration
- Fine-grained delegated write permissions
- API/database layer
- Production authentication/authorization
- Full audit persistence
- Production TLS certificate validation

> This repository is a development prototype and is not production-ready.

## Architecture

```text
User
  |
  v
Local Qwen model
  |
  v
Agent / guardrails
  |
  +------------------------+
  |                        |
  | read                   | mutation
  v                        v
MCP                    Approval Manager
  |                        |
  |                        v
  |                   Human Approval
  |                        |
  +-----------+------------+
              |
              v
       Persistent MCP Server
              |
              v
       DirectoryService
          /        \
         /          \
      Mock          LDAP
                     |
                     v
                  LDAPS
                     |
                     v
                Samba AD
```

## Repository Layout

```text
itsm-agent/
├── agent/
│   ├── agent.py
│   ├── approvals.py
│   └── mcp_client.py
├── services/
│   └── directory/
│       ├── base.py
│       ├── factory.py
│       ├── ldap.py
│       └── mock.py
├── tools/
│   ├── account.py
│   ├── access.py
│   └── registry.py
├── scripts/
│   └── ldap_preflight.py
├── test/
├── docker/
│   └── samba-ad/
│       ├── Dockerfile
│       └── entrypoint.sh
├── config.py
├── main.py
├── mcp_server.py
├── docker-compose.yml
├── requirements.txt
└── .env
```

## Requirements

Recommended development environment:

- Linux or WSL2
- Python 3.12
- Conda or another Python virtual environment
- Docker Engine
- Docker Compose
- Enough RAM to load the local Qwen model

The model is expected to be available locally. The current development model is Qwen3-0.6B.

## Installation

Create or activate your environment:

```bash
conda activate qwen-infra
```

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

Edit `.env` for your environment before starting the LDAP backend.

Never commit `.env`.

## Directory Backends

The project supports two directory backends.

### Mock backend

Useful for agent and approval-flow development without LDAP:

```env
DIRECTORY_BACKEND=mock
```

### LDAP backend

Uses the `LdapDirectoryService` implementation:

```env
DIRECTORY_BACKEND=ldap
```

The LDAP backend supports separate credentials for reads and approved mutations.

```env
AD_BIND_DN=CN=svc_itsm_reader,CN=Users,DC=itsm,DC=local
AD_BIND_PASSWORD=change-me

AD_WRITE_BIND_DN=CN=svc_itsm_writer,CN=Users,DC=itsm,DC=local
AD_WRITE_BIND_PASSWORD=change-me
```

For a real environment, use narrowly delegated service accounts instead of Domain Administrator credentials.

## Local Samba AD Development Environment

The repository contains a Samba Active Directory Domain Controller for local development.

Start it with:

```bash
docker compose up -d samba-ad
```

Check its status:

```bash
docker compose ps
```

The development Compose configuration exposes:

```text
LDAP   localhost:1389 -> container:389
LDAPS  localhost:1636 -> container:636
```

The current Python LDAP configuration uses LDAPS.

To inspect Samba logs:

```bash
docker compose logs -f samba-ad
```

To stop the environment while keeping the AD database:

```bash
docker compose down
```

To completely destroy the local Samba domain and its data:

```bash
docker compose down -v --remove-orphans
```

Use the destructive command only when you intentionally want to reprovision the development domain.

## LDAP Configuration

Example:

```env
DIRECTORY_BACKEND=ldap

AD_HOST=127.0.0.1
AD_PORT=1636
AD_USE_SSL=true

AD_BIND_DN=CN=svc_itsm_reader,CN=Users,DC=itsm,DC=local
AD_BIND_PASSWORD=change-me

AD_WRITE_BIND_DN=CN=svc_itsm_writer,CN=Users,DC=itsm,DC=local
AD_WRITE_BIND_PASSWORD=change-me

AD_BASE_DN=DC=itsm,DC=local
AD_TEST_USER=jdoe

AD_ACCESS_GROUPS={"vpn":"VPN-Users","admin":"ITSM-Admins"}
```

For the throwaway Samba development environment, an Administrator bind may be used temporarily for mutation testing. Do not copy that setup into production.

## LDAP Preflight

Before running the full agent against LDAP, test the directory connection:

```bash
python scripts/ldap_preflight.py
```

A healthy result should include:

```text
[OK] Configuration present
[OK] TCP connection succeeded
[OK] LDAP adapter initialized
[OK] LDAP bind succeeded
[OK] Test account lookup

LDAP read-only preflight passed.
```

## Running the Agent

Start the CLI:

```bash
python main.py
```

Available CLI commands:

```text
approvals
approve <approval_id>
exit
```

Example read request:

```text
is jdoe locked?
```

Example access request:

```text
does jdoe have VPN access?
```

Example mutation request:

```text
unlock asmith
```

A mutation does not execute immediately. The agent creates an approval:

```text
Action 'unlock_user' requires human approval.
Approval ID: abc12345
```

The operator must explicitly approve it:

```text
approve abc12345
```

Do not include quotation marks around the approval ID.

## Tool Policy

Current tools:

| Tool | Type | Approval |
|---|---|---|
| `account_status` | Read | No |
| `check_access` | Read | No |
| `unlock_user` | Mutation | Yes |
| `reset_password` | Mutation | Yes |

`reset_password` remains intentionally disabled in the real LDAP backend.

The language model is not the security boundary. Tool metadata, identifier validation, approval checks, directory permissions, and result verification must remain enforced in application code.

## Active Directory Behavior

`account_status` currently reads:

- `userAccountControl`
- `lockoutTime`

`check_access` currently evaluates direct `memberOf` group membership against the configured resource-to-group mapping.

The initial implementation does not perform recursive/nested-group resolution.

`unlock_user` clears the account lockout state through LDAP only after the existing approval layer authorizes the action.

## Development Safety Notes

- Keep `.env` out of Git.
- Use separate read and mutation service accounts.
- Give write accounts only the permissions they need.
- Do not use production AD credentials in the local development environment.
- Do not disable Samba/AD transport-security requirements merely to simplify client code.
- Treat the current Samba container's `privileged: true` setting as development-only.
- Verify mutation results with a follow-up directory read.
- Keep password reset disabled until password-policy and credential-handling behavior is implemented safely.

## Testing

Compile-check the LDAP adapter:

```bash
python -m py_compile services/directory/ldap.py
```

Run the LDAP preflight:

```bash
python scripts/ldap_preflight.py
```

Then run the complete agent:

```bash
python main.py
```

A useful integration sequence is:

```text
is asmith locked?
unlock asmith
approve <approval_id>
is asmith locked?
```

## Roadmap

Near-term work:

1. Finish and verify the real LDAP account-unlock lifecycle.
2. Add dedicated least-privilege write-service-account permissions.
3. Add automated tests around LDAP attribute parsing and mutation verification.
4. Implement password reset safely behind the existing approval system.
5. Replace prompt-embedded tool descriptions with schema-driven tool metadata.
6. Add persistent approval/audit storage.
7. Add API and authentication layers when the core agent flow is stable.

## License

Licensed under the MIT License. See [LICENSE](LICENSE).
