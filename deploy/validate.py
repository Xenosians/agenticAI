"""Validate the reviewed Compose topology without building or starting it.

Use --no-env-resolution so private runtime env-file contents never enter the
rendered model/output. A successful check establishes configuration only.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import urlsplit


class DeploymentError(ValueError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise DeploymentError(code)


def validate_model(model: dict) -> dict:
    require(type(model) is dict, "invalid_compose_model")
    services = model.get("services", {})
    require(type(services) is dict and all(type(service) is dict for service in services.values()), "invalid_compose_model")
    require(set(services) == {"database", "backend", "frontend", "ai", "schema"}, "service_layout_mismatch")
    for name, service in services.items():
        require(not service.get("privileged") and service.get("network_mode") != "host", "privileged_container_forbidden")
        require(not service.get("cap_add"), "extra_container_capabilities_forbidden")
        require(service.get("restart") == "no" if name == "schema" else service.get("restart") == "unless-stopped", "restart_policy_mismatch")
        require("ALL" in service.get("cap_drop", []), "capability_drop_required")
        require(any(value in {"no-new-privileges:true", "no-new-privileges=true"} for value in service.get("security_opt", [])), "no_new_privileges_required")
        require(name == "frontend" or not service.get("ports"), "internal_port_published")
        for volume in service.get("volumes", []):
            require("docker.sock" not in volume.get("source", "") and "docker.sock" not in volume.get("target", ""), "docker_socket_forbidden")
            if volume.get("type") == "bind":
                require(volume.get("bind", {}).get("create_host_path") is False, "implicit_host_path_creation_forbidden")

    frontend = services["frontend"]
    ports = frontend.get("ports", [])
    require(len(ports) == 1 and ports[0].get("target") == 8443, "frontend_port_layout_mismatch")
    origin = frontend.get("environment", {}).get("PUBLIC_ORIGIN", "")
    host = frontend.get("environment", {}).get("PUBLIC_HOST", "")
    try:
        parsed = urlsplit(origin)
        port = parsed.port if parsed.port is not None else 443
    except ValueError:
        raise DeploymentError("invalid_public_origin") from None
    require(parsed.scheme == "https" and parsed.hostname == host and parsed.path == "" and 1 <= port <= 65535 and
            not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
            "invalid_public_origin")
    require(not origin.endswith(":443"), "public_origin_must_use_browser_canonical_form")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise DeploymentError("callback_requires_certificate_dns_hostname")
    require(str(port) == str(ports[0].get("published")), "public_origin_port_mismatch")
    aliases = frontend.get("networks", {}).get("service", {}).get("aliases", [])
    require(host in aliases, "callback_hostname_alias_required")
    tls = {volume.get("target"): volume for volume in frontend.get("volumes", [])}
    require(all(tls.get(target, {}).get("read_only") is True for target in ("/run/tls/tls.crt", "/run/tls/tls.key")), "tls_files_must_be_read_only")

    backend = services["backend"]
    env = backend.get("environment", {})
    require(env.get("AUTH_COOKIE_SECURE") == "true" and env.get("PHX_BIND_IP") == "0.0.0.0", "production_endpoint_policy_mismatch")
    require(env.get("RELEASE_DISTRIBUTION") == "none" and env.get("RELEASE_TMP") == "/tmp/janus-release", "single_host_release_policy_mismatch")
    require(all(env.get(key) == origin for key in ("PHX_PUBLIC_URL", "APP_FRONTEND_URL", "CORS_ALLOWED_ORIGINS")), "public_origin_binding_mismatch")
    require(env.get("AI_SERVICE_URL") == "http://ai:8000" and env.get("SURREALDB_URL") == "http://database:8000", "internal_service_urls_mismatch")
    require("ai" not in backend.get("depends_on", {}), "startup_dependency_cycle")

    ai = services["ai"]
    ai_env = ai.get("environment", {})
    require(ai_env.get("PHOENIX_BASE_URL") == f"https://{host}:9443", "private_tls_callback_required")
    volumes = {volume.get("target"): volume for volume in ai.get("volumes", [])}
    base_mounts = {"/models", "/state", "/workspace"}
    profile_mounts = {"/run/janus/profiles", "/run/secrets/itsm-links.json"}
    trust_target = "/run/janus/trust/ca-bundle.pem"
    trust_mounts = {trust_target} if trust_target in volumes else set()
    machine_profile = bool(set(volumes) & profile_mounts)
    require(set(volumes) == base_mounts | (profile_mounts if machine_profile else set()) | trust_mounts, "ai_mount_layout_mismatch")
    if trust_mounts:
        require(volumes[trust_target].get("read_only") is True and tls.get(trust_target, {}).get("read_only") is True and
                volumes[trust_target].get("source") == tls[trust_target].get("source") and
                ai_env.get("ITSM_HTTP_CA_BUNDLE") == trust_target and
                frontend.get("environment", {}).get("CURL_CA_BUNDLE") == trust_target,
                "tls_trust_binding_invalid")
    else:
        require(trust_target not in tls and "ITSM_HTTP_CA_BUNDLE" not in ai_env and
                "CURL_CA_BUNDLE" not in frontend.get("environment", {}), "tls_trust_mounts_missing")
    if machine_profile:
        require(all(volumes[target].get("read_only") is True for target in profile_mounts), "lua_profile_mounts_must_be_read_only")
        require(ai_env.get("ITSM_LUA_ADAPTER_FILE") == "/run/janus/profiles/machine.lua" and
                ai_env.get("ITSM_LINK_SECRETS_FILE") == "/run/secrets/itsm-links.json" and
                re.fullmatch(r"[a-f0-9]{64}", ai_env.get("ITSM_LUA_ADAPTER_SHA256", "")) is not None,
                "lua_profile_binding_invalid")
    else:
        require(not any(key in ai_env for key in ("ITSM_LUA_ADAPTER_FILE", "ITSM_LUA_ADAPTER_SHA256", "ITSM_LINK_SECRETS_FILE")),
                "lua_profile_mounts_missing")
    require(volumes["/models"].get("read_only") is True and not volumes["/state"].get("read_only") and not volumes["/workspace"].get("read_only"), "ai_storage_modes_mismatch")
    require(ai.get("read_only") is True and ai.get("deploy", {}).get("replicas", 1) == 1, "single_ai_process_storage_required")
    for key in ("COMPLETION_OUTBOX_PATH", "APPROVAL_STORE_PATH", "EXTENSION_STORAGE_ROOT", "USER_MODEL_ROOT", "MODEL_ARTIFACT_CACHE_ROOT",
                "CONTINUAL_STATE_DB", "LEARNING_TRAJECTORY_PATH", "LEARNING_CORRECTION_PATH", "LEARNING_CONTEXT_PATH", "LEARNING_RUN_ROOT", "MAILBOX_PROVISIONING_LOCK_PATH"):
        path = PurePosixPath(ai_env.get(key, ""))
        require(path.is_absolute() and path.parts[:2] == ("/", "state") and len(path.parts) > 2 and ".." not in path.parts,
                "durable_ai_state_not_persisted")
    require(ai_env.get("HF_HUB_OFFLINE") == "1" and ai_env.get("TRANSFORMERS_OFFLINE") == "1", "unrequested_model_downloads_forbidden")
    schema = services["schema"]
    require(schema.get("profiles") == ["bootstrap"] and schema.get("command") == ["schema"], "schema_bootstrap_must_be_explicit")
    require("schema" not in {dependency for service in services.values() for dependency in service.get("depends_on", {})}, "automatic_schema_bootstrap_forbidden")
    require(schema.get("environment") == env, "schema_runtime_configuration_mismatch")
    return {"status": "configuration_validated", "services": sorted(services),
            "images_built": False, "runtime_started": False, "e2e_verified": False}


def render_compose(env_file: Path, *, gpu: bool = False, machine_profile: bool = False, private_ca: bool = False) -> dict:
    root = Path(__file__).resolve().parent
    command = ["docker", "compose", "--env-file", str(env_file.resolve()), "-f", str(root / "compose.yaml")]
    if gpu:
        command += ["-f", str(root / "compose.gpu.yaml")]
    if machine_profile:
        command += ["-f", str(root / "compose.profile.yaml")]
    if private_ca:
        command += ["-f", str(root / "compose.trust.yaml")]
    command += ["--profile", "bootstrap", "config", "--no-env-resolution", "--format", "json"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise DeploymentError("compose_cli_unavailable_or_timed_out") from None
    require(completed.returncode == 0, "compose_configuration_rejected")
    try:
        model = json.loads(completed.stdout)
    except (ValueError, TypeError):
        raise DeploymentError("invalid_compose_model") from None
    require(type(model) is dict, "invalid_compose_model")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--machine-profile", action="store_true")
    parser.add_argument("--private-ca", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(validate_model(render_compose(args.env_file, gpu=args.gpu,
                                                     machine_profile=args.machine_profile, private_ca=args.private_ca))))
        return 0
    except DeploymentError as error:
        print(json.dumps({"status": "configuration_rejected", "code": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
