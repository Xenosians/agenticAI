# Extension stores and runtime contracts

This note describes the current store surfaces and the next safe steps for
MCP tools and Lua plugins. It is an implementation guide, not a claim that
user-authored plugins can already be installed or activated.

## Store boundaries

The browser shows the Model Store on its own and one **Store** page with three
sections: Tools, Skill Library and Plugin Store.

| Store | Current source of truth | What the UI may claim |
| --- | --- | --- |
| Model Store | Authenticated extension catalog: owner-selected model connections, local model profiles, harness choices and datasets | A saved resource exists; selection and validation status come from the returned resource metadata. A listed provider is not proof that it is reachable. |
| Skill Library (Store section) | The owner's versioned skill library | A skill is saved, enabled, disabled or versioned. Skill text can guide written answers; it cannot select a tool or grant permission. |
| Tools (Store section) | `tools.registry.TOOLS`, checked against the actual MCP runtime registration and the same capability contracts used by the orchestrator | A capability is registered and its declared effect, risk, provider and approval requirements. Registration is not provider health or evidence of a completed action. |
| Plugin Store (Store section) | The runtime's `PLUGIN_FILES` inventory, resolved through the active reviewed machine profile | An adapter source resolves for its protocol and has a specific SHA-256 identity. Resolution is not proof of successful execution, user ownership or provider availability. |

The authenticated browser reads the catalog through Phoenix. Phoenix derives
the owner from the signed-in session and calls the private AI endpoint; a
request field cannot choose a different owner. Catalog responses omit
credentials and machine filesystem paths. The Model, Skill and Plugin views
share the existing extension catalog request. Provider readiness remains a
separate observation.

## Execution and authority

1. MCP tools remain declared in the AI tool registry and registered by the MCP
   server. Registry parity and typed capability checks remain build/runtime
   gates. A catalog page cannot register or enable a new executor.
2. Every model, connection, source and tool dispatch crosses the required Lua
   adapter. The host pins the plugin ID and source digest, sends bounded
   credential-free data to the interpreter, validates the result, then performs
   the admitted network, model or tool operation.
3. Phoenix owns identity, authorization and durable extension lifecycle. Lua
   may map protocol data; it cannot create a permission, rewrite the approved
   target, raise a budget, bypass confirmation or certify a provider effect.
4. Provider and MCP receipts remain authoritative for success. An adapter
   resolving or returning a mapped payload is only an adapter result. A failure
   after a real mutation preserves the actual receipt and requires
   reconciliation when the effect is uncertain.

## Current scope

The Model and Skill stores manage the owner's own resources; the MCP page
adds a per-owner on/off switch for each tool (below); the Lua page is a
read-only catalog. The MCP page consumes the catalog already generated from
`workflow_capabilities` and reports registration metadata, never provider
health. The Lua page resolves each packaged or machine-profile adapter and
reports its state and digest without returning its path. A broken selected
adapter is shown as unavailable; the catalog does not substitute the packaged
adapter. The existing model and skill management flows keep their current
owner and validation behavior.

Phoenix keeps owner-scoped Lua extension records (`/api/v1/extensions/records`):
a manifest validated against `extension-manifest.v1` with its digest pinned,
a lifecycle `pending` → `reviewed` → `active` ⇄ `disabled`, `revoked` from any
state after review, and an audit event for every transition. Those records
store no Lua source bytes, and dispatch does not read them yet, so a record
changes nothing that runs. The current machine profile still selects reviewed
Lua files globally; it is not an owner-scoped plugin registry. The store UI
must remain explicit about this.

## Requested store experience

The product direction is one discoverable store family with an item state,
source provenance and an add flow. User-managed entries need an owner-scoped,
durable on/off control. Turning an entry off removes it from future selection
or dispatch while retaining its version and audit history; it does not change
an operation already in flight. Any required system adapter that is disabled
or unavailable must stop the affected operation. There is no non-Lua fallback.

Seed the stores from real built-in sources: the MCP runtime registry,
packaged/reviewed Lua adapters, enabled local model profiles, the reviewed
built-in skills below, and each owner's actual saved resources and skills.
Mark system-owned entries separately from user entries. Do not create fake
provider connections, sample credentials, or placeholder skills to make a
store look populated. A seeded entry's visible readiness still comes from a
real readiness observation.

Add methods are source-specific because these stores admit different kinds of
data:

| Store | Current/add source direction | Admission and authority boundary |
| --- | --- | --- |
| Model | Provider API URL, uploaded local model bundle, or create a connection in the app | Existing connection validation and model-bundle format checks; never treat a saved URL as reachable. |
| Dataset | Text/document upload, reviewed JSONL upload, or a source URL/crawl | Existing purpose-specific extraction, Lua admission and dataset review/version controls. ZIP support requires a declared archive format and bounded extraction. |
| Skill | Create/edit in the app, upload `SKILL.md`, or import a ZIP/URL containing skill files | Parse front matter, size and sensitive-content checks for every skill; imports create immutable versions and do not grant tool authority. |
| MCP Tool | Built-in runtime capability; a future remote MCP connection can be added by URL | A URL registers a typed, reviewed server connection. It cannot inject Python, Lua or arbitrary tools into the host registry. New capabilities still need host executors and policy contracts. |
| Lua Plugin | Create/edit Lua in the app, upload a `.lua` file or declared ZIP, or import a pinned source URL | Validate manifest/protocol, exact digest, source bounds and sandbox contract before review. URL fetch must block private/link-local destinations and revalidate redirects/DNS to prevent SSRF. No browser-supplied filesystem path. |

The frontend should show source type, owner/system provenance, version or
digest, validation/review state and the enable switch from API data. A switch
must call a durable backend transition and refresh authoritative state; it
must not optimistically claim success. The app editor is the native create
flow. ZIP and URL intake are enabled only for the store types whose validators
and controlled ingestion path exist.

Current features should remain clearly separated from planned imports: the
Model Store already supports API connections and local model uploads; datasets
support their current document/JSONL and crawl flows; the Skill Library already
supports app editing, `SKILL.md` upload, versions and toggling; the Tools section
groups tools into packages and has an owner on/off switch per package and per
tool (below). The Plugin Store is a read-only runtime catalog; the Phoenix record lifecycle above has no store
screen yet. ZIP intake, user Lua URLs/editor, a Lua bytes store and remote MCP
registration are not yet implemented.

### Owner tool switches

An owner can turn a registered MCP tool off for their own account (action
`tool_enable` with `{"id", "enabled"}`; the catalog's `capabilities` rows carry
`enabled`). `services/extensions/tool_switches.py` stores only the turned-off
tools per owner (no row means on); unregistered names and non-boolean values
are refused (`tool_not_found`, `tool_enabled_invalid`). Each request reads its
owner's switches once into `disabled_tools`; `ToolGateway` refuses a turned-off
tool for that request with `denied` / `tool_disabled_by_owner` before grounding,
policy, approval or provider, so no proposal made before the switch can slip
through. The switch is a preference, not an authority: turning a tool on grants
nothing its policy, approval and provider checks do not already allow. Learning
records the outcome as the owner's choice (`owner_disabled_tool`), never as a
model failure.

Tools are grouped into **packages**, one per provider family of
`services/integration_readiness.py` (every registered tool belongs to exactly
one). The catalog's `tool_packages` rows carry each package's `id`, `name`,
`kind` and setup `state` from the same readiness rule as the integration status;
each capability row's `provider` names its package. Action
`tool_package_enable` with `{"package", "enabled"}` switches every tool of one
package in one write (`tool_package_not_found`, `tool_enabled_invalid`); a single
tool can still be switched on its own afterwards. Packaged Lua protocol adapters are required gates and stay
host-owned; owner Lua plugins wait for the reviewed bytes store.

### Built-in skills

`skills/defaults/*.md` holds reviewed skills every owner starts with:
`itsm-gap-finder`, `skill-finder`, `chat-tightening`, `proxmox-operator`,
`tailscale-operator`, `research-with-citations` and `senior-engineer`. Each
file passes the same parser, size and sensitive-content checks as an owner's
skill; a bad file fails loudly. `SkillLibrary` gives an owner each built-in
skill once, on the first read of their library, and records that in
`skill_builtin`. From then on the skill is the owner's: an edit becomes their
next version, turning it off keeps it off, and a deleted built-in skill is not
added again. An owner skill that already had a built-in's name is kept as the
owner's own. Built-in skills are marked `builtin` in the catalog (the Skill
Store shows "Built in") and do not count against the owner's skill limit.
Like every skill they only guide how answers are written; they choose no
tool, grant no access and skip no approval. A new built-in skill reaches
existing owners on their next read; a changed built-in file does not replace
a version an owner already has.

## Next implementation stages

1. Define a versioned, language-neutral plugin manifest with a stable ID,
   protocol, version, declared inputs/outputs, resource limits and exact source
   digest. Include source provenance and required/optional status. Reject
   unknown fields and unsupported protocols. Never accept an executable path
   from a browser.
2. Phoenix-owned, owner-scoped lifecycle records and audit events (done; see
   Current scope). A plugin must pass validation and explicit review before
   activation. Revoke is a durable state transition; once dispatch reads the
   records it must prevent future dispatches without rewriting the identity
   pinned to an already-running request.
3. Store immutable Lua bytes in a host-controlled directory or object store.
   Enforce source-size, interpreter-time and output-size limits. Do not expose
   filesystem access, `require`, network, credentials, process creation or host
   authority inside the Lua worker.
4. Resolve active plugins by owner, protocol, version and digest before
   dispatch. Cache only against that complete identity. Missing, revoked,
   malformed or changed artifacts stop dispatch; there is no Python-only or
   packaged-plugin fallback after explicit selection.
5. Add FE source-specific intake and on/off/review/activate/revoke actions only
   after backend persistence and the AI lookup path exist. Every mutation needs
   authenticated CSRF protection, immutable identity, idempotency and a
   read-back of its terminal state. ZIP readers must reject path traversal,
   links, excessive entries and expansion; URL intake must defend against
   private-address SSRF and redirect rebinding.
6. Permit new MCP tool behavior only through a declared capability contract
   and an existing host executor/policy boundary. Lua plugins may not add an
   arbitrary function to the MCP server or bypass SemanticGuard, approvals or
   ToolGateway.

## Required acceptance evidence

- MCP catalog IDs equal the registered runtime tool IDs; local and outbound
  requests see the same capability contracts.
- Owner A cannot list, inspect, activate or revoke Owner B's plugin records.
- Invalid manifest versions, unknown fields, digest mismatches, path traversal,
  symlinks, oversized source and malformed Lua output are rejected.
- A revoked or unavailable adapter blocks future dispatch without silently
  selecting another adapter. An in-flight mutation keeps its exact pinned
  identity and actual provider receipt.
- Search/filter UI is derived from API responses; empty, invalid-schema and
  unavailable states are distinct from a healthy empty catalog.
- Built-in entries are derived from the runtime registry/profile; no fabricated
  defaults appear. Owner A cannot toggle or mutate Owner B's entries. Toggles
  survive refresh/relogin and disabled entries are absent from future
  selection/dispatch, while in-flight operations retain their pinned digest.
- File, ZIP, URL and app-editor intake each have positive and negative cases
  for the declared store type, including malformed manifests, source limits,
  traversal/link archives, disallowed destinations, redirect/DNS changes and
  digest mismatch.
- Provider reachability, adapter resolution, MCP registration, approval and
  real provider completion are reported as separate states.
