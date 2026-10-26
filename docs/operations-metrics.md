# Job observations

During a durable job scope, the existing inference coordinator records bounded,
content-free observations for each model call. Phoenix binds the immutable event
to that job and attempt, derives its owner, and persists it independently of job
outcome. Local measurements come from the actual scheduler and generation;
outbound calls retain measured duration and available budget accounting, with
unavailable GPU/token measurements left null.

The Operations surface reads owner-scoped jobs and these recorded calls. It does
not expose prompts, responses, model paths, credentials, raw logs or hidden
reasoning. Older jobs have no observations and are shown as not recorded.

This is best-effort observability: a bounded queue, unavailable transport or
process interruption may leave gaps. Call durations may overlap and must not be
presented as a complete end-to-end execution time. A measurement is not evidence
that a tool or provider completed work. Failed telemetry cannot change inference,
approval, cancellation or job outcome.

The authenticated internal metrics endpoint accepts retries only under the same
immutable event identity; a changed replay or stale attempt is rejected. The AI
sender uses at most two transport attempts, no redirects or environment proxy,
and a bounded final flush. Existing WebSocket hints and durable completion remain
their own protocols. Standalone evaluation/training without a job context does
not publish job observations.

`JOB_METRICS_ENABLED=false` disables publication for deployments that do not
want these observations. It also isolates offline application fixtures from any
running backend; focused metrics tests inject their own sender. Disabling the
optional progress WebSocket does not disable HTTP metric publication.
