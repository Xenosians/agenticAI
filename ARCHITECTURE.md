# Architecture

## Overview

ITSM Agent separates natural-language reasoning from privileged directory execution.

```text
User request
    |
    v
Local Qwen
    |
    v
Agent
    |
    +---- identifier validation
    |
    +---- tool policy
    |
    +---- read --------------------------+
    |                                    |
    +---- mutation -> approval -> human -+
                                         |
                                         v
                                Persistent MCP Client
                                         |
                                         v
                                     MCP Server
                                         |
                                         v
                                DirectoryService API
                                    /          \
                                   /            \
                             Mock backend     LDAP backend
                                                |
                                                v
                                              LDAPS
                                                |
                                                v
                                             Samba AD
```

## Main Components

### Local model

The current model is Qwen3-0.6B loaded locally.

The model proposes structured actions using the current text protocol:

```text
ACTION: <tool>
ARGS: {...}
```

or returns a final answer.

Model output is not treated as authorization.

### Agent

The agent:

- builds the model prompt
- parses proposed tool calls
- validates identifiers
- reads tool metadata
- decides whether approval is required
- sends permitted calls through MCP
- returns tool results to the user

### Approval manager

Mutation requests are converted into pending approval records.

The tool is executed only after an explicit:

```text
approve <approval_id>
```

A mutation is considered successfully approved/executed only when the tool result confirms successful execution.

### MCP

The MCP server is a separate persistent subprocess.

Both reads and approved mutations cross the MCP boundary.

This keeps the agent/orchestrator separate from the directory-service implementation.

### DirectoryService

`DirectoryService` is the backend abstraction used by the MCP tools.

Current implementations:

- `MockDirectoryService`
- `LdapDirectoryService`

The factory selects the implementation using:

```env
DIRECTORY_BACKEND=mock
```

or:

```env
DIRECTORY_BACKEND=ldap
```

### LDAP backend

The LDAP backend currently supports:

- account status
- access/group checks
- experimental account unlock

Password reset remains disabled.

The backend supports separate read and write bind credentials.

## Security Boundaries

The intended enforcement order for a mutation is:

```text
LLM proposes action
        |
        v
Identifier validation
        |
        v
Tool policy
        |
        v
Approval required
        |
        v
Human approves
        |
        v
MCP mutation
        |
        v
Directory permissions
        |
        v
Post-mutation verification
```

No single layer should be treated as the only security control.

## Current Limitations

- CLI-only interface
- in-memory approvals
- no durable audit database
- direct-group access checks only
- local development Samba AD
- self-signed development TLS
- unlock integration still under active testing
- password reset disabled
- no production identity/authentication layer
