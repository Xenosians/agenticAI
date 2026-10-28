# Single-host deployment foundation

This folder defines the application deployment layout for the three sibling
repositories. It is separate from the existing SurrealDB/Samba/mailbox lab
Compose files. It does not start containers, acquire weights, alter a serving
profile, initialize a database or configure Windows/WSL by being present.

## Layout

```text
agenticaiPersonal/itsm-agent/
  Dockerfile                    Python AI runtime and one Uvicorn worker
  docker/entrypoint.py           preserves normal application startup
  docker/healthcheck.py          validates the actual /ready response
  requirements-container.txt    API additions to existing dependencies
  deploy/
    compose.yaml                database, backend, frontend, AI and explicit schema task
    compose.gpu.yaml            optional existing-GPU-runtime reservation
    *.env.example              deployment and separate private runtime inputs
    validate.py                configuration-only check; never calls compose up
    tests/test_deployment.py    rejection, context and rendering checks
    kubernetes/README.md        conditional adoption criteria
agenticBackend/itsm_backend/
  Dockerfile                    production Phoenix release
  docker/entrypoint.sh          start or explicit schema command
  docker/healthcheck.sh         Phoenix liveness, not AI/provider acceptance
agenticFrontend/
  Dockerfile                    compiles Nim/Karax, then serves static assets
  docker/entrypoint.sh          validates/generates public browser configuration
  docker/nginx.conf.template    HTTPS, REST and browser/internal WebSockets
```

The stack's only published port is frontend HTTPS, bound to loopback by default.
The backend, database, AI endpoint and private callback listener stay on the
project network. No Docker socket, host network or privileged container is
provided. Network egress remains necessary for explicitly configured model,
Jira and mail providers; a private container network is not an egress policy.

## Inputs to prepare

1. Place the sibling checkouts at their documented workspace paths, or override
   `AI_CONTEXT`, `BACKEND_CONTEXT` and `FRONTEND_CONTEXT` explicitly.
2. Copy `.env.example` to an ignored private deployment file. Fill the image
   references with reviewed versions/digests for the target architecture. No
   floating registry default is selected by these files. The image contracts are:
   - Python 3.12 on Debian bookworm, or a compatible reviewed CUDA/PyTorch base;
   - an Elixir/OTP builder and Debian runtime on the same Debian release and
     architecture, because release native libraries must match;
   - a Nim builder containing Nim/Nimble compatible with the repository and Git;
   - an official Alpine-based nginx runtime and a compatible SurrealDB 3 image.
   Application/transitive dependency locking and actual image-build acceptance
   remain required before a release. API dependency pins match the inspected
   runtime; `requirements-container.txt` is not a complete lock.
3. Populate separate private `ai.env`, `backend.env` and `database.env` files from
   their examples. Never copy a host `.env` blindly. Container provider hosts,
   paths, timeouts, logical model keys, endpoint allowlists and tool policies must
   be reviewed for that deployment. Internal service tokens must agree. Database
   credentials must agree between its own file and backend configuration; the
   database does not receive mail or model-provider keys.
4. Provide an existing trusted TLS certificate chain/private key for a DNS
   hostname you control. `PUBLIC_ORIGIN` must exactly match the browser origin
   and published port, for example `https://itsm.example:8443` (illustrative only).
   Use a lowercase DNS hostname; omit `:443` when using the default HTTPS port.
   The same hostname is a service-network alias for HTTPS callbacks on private
   port 9443. Both browser and AI must trust the certificate. The foundation
   expects a publicly trusted chain; it does not disable verification or install
   a private CA into the host or dependencies. TLS files mounted into the
   frontend must be readable by its UID/GID **101:101**, including its private key.
   Prepare appropriately protected deployment copies; do not make a system key
   world-readable to work around permissions.
5. Pre-create dedicated database/AI state/tool-workspace directories. Bind mounts
   refuse implicit host directory creation. Database and AI use UID/GID
   **10001:10001** and need the intended read/write access; acquired weights are
   mounted read-only under `/models`. The approved workspace is the only host
   tool workspace mounted at `/workspace`. No startup script changes host owners
   or permissions. Do not point a test stack at existing live data directories.

Secrets are runtime env-file inputs, not build arguments or copied image files.
Docker administrators can inspect container environment variables. Protect the
host files and daemon access; this layout is not a secret-vault implementation.
The ignore policies exclude nested env/key/PEM files as well as known runtime,
model and deployment-private folders. Explicit safe `*.env.example` files are
retained. Never add a secret to an example file.

## Local and outbound inference

Both use the same AI service and execution boundary. Its current startup warms
the configured Hub. Choosing an outbound connection for an account does not
remove that prerequisite or prove the AI can boot without its configured local
profiles. Supply acquired weights and valid profile definitions with container
paths such as `/models/...` when those profiles are local. The Dockerfiles and
Compose file never change `MODEL_PROFILES`, select an account connection, replace
the networking specialist or silently substitute a mock provider.

Model downloads are disabled by default. Uploaded user models, encrypted extension
metadata/key, approvals, completion outbox and model caches live under the AI's
dedicated `/state` mount. Continual-control SQLite, capture ledgers and mailbox
provisioning locks use that mount too. The control store is initialized even
when learning is disabled; persisting it does not enable training. An explicitly
selected SQLite asset provider also needs an explicit `/state/...` database path.
Training dispatch/artifact paths need separate review before enabling candidate
jobs in a container; these files do not provide that integration.
Run one AI worker per mutable storage root. A second
container sharing that root is not supported by this foundation.

The optional GPU overlay reserves one GPU through an already configured compatible
runtime. It installs no driver/toolkit and does not alter memory budgets. Review
the base image, PyTorch build and host compatibility first. CPU-only configuration
validation is not proof the configured local models can run on CPU or fit a GPU.

Shell/Git/provider readiness is still deployment-specific: configure logical
repositories and the command allowlist for `/workspace`, provide reviewed build
tools when needed, and set actual reachable provider endpoints. `127.0.0.1` inside
a container means that container. Existing lab providers are not implicitly
attached, enabled or reconfigured. Container-operation tools need a separately
reviewed provider endpoint; no blanket Docker socket is exposed.

## Validate, then deliberately build/bootstrap/start

From the AI checkout, configuration validation uses Docker Compose without
building, starting or reading the private runtime env files into its output:

```sh
python deploy/validate.py --env-file /absolute/path/deployment.private.env
```

Use `--gpu` to validate the optional overlay too. A successful result says
`configuration_validated`, `images_built: false`, `runtime_started: false` and
`e2e_verified: false`. It does not establish certificate validity, provider
connectivity, usable weights or a passing deployment.

When the operator's reviewed inputs are ready, the deliberate deployment sequence
is below. These commands are documentation, not actions already performed:

```sh
docker compose --env-file /absolute/path/deployment.private.env -f deploy/compose.yaml build
docker compose --env-file /absolute/path/deployment.private.env -f deploy/compose.yaml up -d database
docker compose --env-file /absolute/path/deployment.private.env -f deploy/compose.yaml --profile bootstrap run --rm schema
docker compose --env-file /absolute/path/deployment.private.env -f deploy/compose.yaml up -d backend frontend ai
```

For a reviewed GPU deployment, add `-f deploy/compose.gpu.yaml` to each relevant
command. Do not start the normal services until explicit schema bootstrap passes.
The schema task starts Req only and calls the existing canonical schema helper;
it does not start application workers, dispatch jobs or send mail.

The backend remains a production release with secure cookies, the exact public
origin/CORS policy and HTTPS forwarding. `MAILER_MODE=local` remains prohibited
by production configuration. Provide the configured real mail transport rather
than weakening those checks. That production mode also leaves development-only
lab mail routing disabled. Provider acceptance/inbox delivery require their own
correlated acceptance evidence.

Single-host releases use `RELEASE_DISTRIBUTION=none`; automatic Erlang node
distribution is not needed here. `RELEASE_TMP=/tmp/janus-release` keeps release
runtime-configuration files on the writable tmpfs rather than the read-only
application filesystem. Future distributed deployment needs a reviewed network
and runtime cookie policy; an image-generated release cookie is not a production
credential to share across nodes.

The frontend terminates TLS and preserves WebSocket upgrade headers. Public
requests cannot route to `/api/internal/` or `/ai_socket`; AI callbacks and the
AI status socket use the separate non-published TLS listener. That listener also
proxies the exact `/api/health` read used by Operations, while refusing other
public API paths. Internal token,
owner, job and attempt checks remain Phoenix application controls.

## Verification and recovery gates

Run the offline checks from the AI repository:

```sh
python -B deploy/tests/test_deployment.py
```

They validate Compose with synthetic inputs, verify negative policy cases,
exercise health/entrypoint response handling and use Docker's real build-context
filter against dummy secret files in a scratch export. No application image,
model, provider, mail recipient or application database is involved. If required
tools are absent, skips remain explicit and cannot be counted as deployment proof.

Still required before deployment acceptance:

- Build the real application images from pinned reviewed inputs, then scan and
  record exact image/dependency/source identities.
- Verify real TLS, schema initialization, auth/chat, uploads/retrieval and both
  local/outbound tool paths inside the isolated stack.
- Prove browser and AI WebSocket reconnect, in-flight approvals, terminal state
  and process restart preserve durable identities without repeated effects.
- Back up the database, AI state, encrypted extension metadata **and its
  credentials.key**, approved workspace and model/artifact identities. Test a
  restore in separate directories before relying on a backup.
- Roll back images/configuration only with compatible persistent schema and
  verified artifacts. Never delete approvals/outboxes or automatically resend an
  uncertain task/mail/report to make a deployment look healthy.

Health checks measure service liveness/readiness. They are not the 84-tool
acceptance gate. Kubernetes adoption remains conditional in its companion guide.

References: [Compose application model](https://docs.docker.com/compose/intro/compose-application-model/),
[Mix releases and one-off eval](https://hexdocs.pm/mix/Mix.Tasks.Release.html),
[SurrealDB readiness CLI](https://surrealdb.com/docs/surrealdb/cli/isready).
