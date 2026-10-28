# Conditional Kubernetes adoption

There is no cluster installation or deployment in this foundation. The immediate
target is a verified single-host Compose stack using the sibling application
images. Reuse those accepted images when a real multi-host, independent scaling
or host-failure availability requirement justifies cluster operation.

Before adding executable manifests, decide and verify:

1. Image registry/digest identities, ingress/TLS and exact public origin, secret
   distribution and scoped provider networking.
2. Persistent storage/access modes, database operations and an independently
   tested backup/restore procedure. New replicas do not automatically make a
   stateful database highly available.
3. Startup/readiness/liveness probes, graceful termination, resource requests,
   measured GPU allocation and rollout/rollback behavior.
4. AI state coordination. Keep one worker/pod per mutable approval/outbox/model/
   extension storage root until multi-worker coordination is implemented and
   tested. Multiple GPU replicas also consume their own model memory.
5. Durable job ownership, exact approvals, lease recovery and unknown-outcome
   reconciliation during restart/failover. An orchestrator retry cannot authorize
   another provider mutation or mail delivery.
6. An isolated cluster acceptance run covering the real local/outbound tools,
   authenticated browser/AI WebSockets and storage recovery. Manifest syntax and
   a healthy Pod alone do not establish those outcomes.

Choose Helm or Kustomize after these contracts are concrete. No placeholder
manifest is presented as a working cluster, and no host/WSL/driver setting is
changed here. See the [Kubernetes overview](https://kubernetes.io/docs/concepts/overview/)
for its deployment/scaling/failover responsibilities.
