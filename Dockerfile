# syntax=docker/dockerfile:1
ARG PYTHON_RUNTIME_IMAGE
# Image contract: a reviewed Python 3.12 Debian-bookworm image, optionally with
# the operator's compatible CUDA/PyTorch runtime. No weights enter the image.
FROM ${PYTHON_RUNTIME_IMAGE} AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git openssh-client build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 janus && useradd --uid 10001 --gid janus --create-home janus
WORKDIR /app
COPY requirements.txt requirements-container.txt ./
RUN python -m pip install --no-cache-dir -r requirements-container.txt
COPY --chown=10001:10001 . ./
RUN mkdir -p /state /workspace && chown 10001:10001 /state /workspace
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    EXTENSION_STORAGE_ROOT=/state/extensions USER_MODEL_ROOT=/state/user-models \
    COMPLETION_OUTBOX_PATH=/state/completion_outbox.sqlite3 APPROVAL_STORE_PATH=/state/approvals.sqlite3 \
    MODEL_ARTIFACT_CACHE_ROOT=/state/model-cache PROCESS_WORKSPACE_ROOT=/workspace \
    CONTINUAL_STATE_DB=/state/learning/continuous/state.sqlite3 \
    LEARNING_TRAJECTORY_PATH=/state/learning/trajectories.jsonl \
    LEARNING_CORRECTION_PATH=/state/learning/corrections.jsonl \
    MAILBOX_PROVISIONING_LOCK_PATH=/state/mailbox-provisioning.lock
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["python", "docker/entrypoint.py"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 CMD ["python", "docker/healthcheck.py"]
