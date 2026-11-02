"""Offline deployment contracts. No application images/services are started.

Docker's context test builds only a FROM scratch/COPY export from dummy files.
Its fixture contains no deployment settings, weights or real credentials.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


DEPLOY = Path(__file__).resolve().parents[1]
AI = DEPLOY.parent
WORKSPACE = DEPLOY.parents[2]
BACKEND = WORKSPACE / "agenticBackend/itsm_backend"
FRONTEND = WORKSPACE / "agenticFrontend"
OUT = WORKSPACE / "tmp/deployment-foundation-20261028"
OUT.mkdir(parents=True, exist_ok=True)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


validation = module(DEPLOY / "validate.py", "deployment_validation")


class ComposeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("docker"):
            raise unittest.SkipTest("Docker Compose is unavailable; configuration checks did not run")
        cls.temp = tempfile.TemporaryDirectory(dir=OUT)
        cls.root = Path(cls.temp.name)
        values = {
            "COMPOSE_PROJECT_NAME": "janus-foundation-fixture",
            "AGENTIC_CONTEXT": str(AI), "BACKEND_CONTEXT": str(BACKEND), "FRONTEND_CONTEXT": str(FRONTEND),
            "PYTHON_RUNTIME_IMAGE": "python:fixture@sha256:" + "1" * 64,
            "ELIXIR_BUILD_IMAGE": "elixir:fixture@sha256:" + "2" * 64,
            "DEBIAN_RUNTIME_IMAGE": "debian:fixture@sha256:" + "3" * 64,
            "NIM_BUILD_IMAGE": "nim:fixture@sha256:" + "4" * 64,
            "NGINX_RUNTIME_IMAGE": "nginx:fixture@sha256:" + "5" * 64,
            "SURREALDB_IMAGE": "surrealdb:fixture@sha256:" + "6" * 64,
            "AGENTIC_IMAGE": "agentic:fixture", "BACKEND_IMAGE": "backend:fixture", "FRONTEND_IMAGE": "frontend:fixture",
            "PUBLIC_HOST": "itsm.example.test", "PUBLIC_ORIGIN": "https://itsm.example.test:8443",
            "PUBLIC_TLS_PORT": "8443", "PUBLIC_BIND_IP": "127.0.0.1",
        }
        for key in ("AGENTIC_ENV_FILE", "BACKEND_ENV_FILE", "DATABASE_ENV_FILE", "TLS_CERT_FILE", "TLS_KEY_FILE"):
            path = cls.root / key.lower()
            path.write_text("FOUNDATION_PRIVATE_MARKER=not-a-real-credential\n")
            values[key] = str(path)
        for key in ("MODELS_ROOT", "AGENTIC_STATE_ROOT", "DATABASE_DATA_ROOT", "WORKSPACE_ROOT"):
            path = cls.root / key.lower()
            path.mkdir()
            values[key] = str(path)
        profile_root = cls.root / "profile"
        profile_root.mkdir()
        (profile_root / "machine.lua").write_text("return {links = {}}\n")
        values["LUA_PROFILE_ROOT"] = str(profile_root)
        values["LUA_PROFILE_SHA256"] = hashlib.sha256((profile_root / "machine.lua").read_bytes()).hexdigest()
        values["LUA_LINK_SECRETS_FILE"] = str(cls.root / "fixture-link-secrets.json")
        Path(values["LUA_LINK_SECRETS_FILE"]).write_text("{}\n")
        values["TLS_CA_BUNDLE_FILE"] = str(cls.root / "fixture-ca-bundle.pem")
        Path(values["TLS_CA_BUNDLE_FILE"]).write_text("fixture-marker-not-a-certificate\n")
        cls.env = cls.root / "deployment.env"
        cls.env.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n")
        cls.model = validation.render_compose(cls.env)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temp"):
            cls.temp.cleanup()

    def reject(self, change, code):
        model = copy.deepcopy(self.model)
        change(model["services"])
        with self.assertRaisesRegex(validation.DeploymentError, code):
            validation.validate_model(model)

    def test_real_compose_parser_preserves_the_private_single_host_layout(self):
        report = validation.validate_model(self.model)
        self.assertEqual(report["status"], "configuration_validated")
        self.assertFalse(report["images_built"])
        self.assertFalse(report["runtime_started"])
        self.assertFalse(report["e2e_verified"])
        self.assertEqual(self.model["services"]["frontend"]["ports"][0]["host_ip"], "127.0.0.1")

    def test_render_does_not_resolve_private_runtime_env_files(self):
        self.assertNotIn("FOUNDATION_PRIVATE_MARKER", json.dumps(self.model))
        self.assertIn("env_file", self.model["services"]["agentic"])

    def test_each_application_image_is_named_after_its_service(self):
        images = {name: service["image"] for name, service in self.model["services"].items() if name != "database"}
        self.assertEqual(images, {"agentic": "agentic:fixture", "backend": "backend:fixture",
                                  "frontend": "frontend:fixture", "schema": "backend:fixture"})

    def test_a_deployment_file_with_the_old_ai_names_is_refused(self):
        old = self.root / "old-names.env"
        old.write_text(self.env.read_text().replace("AGENTIC_", "AI_"))
        with self.assertRaisesRegex(validation.DeploymentError, "compose_configuration_rejected"):
            validation.render_compose(old)

    def test_gpu_is_an_explicit_overlay_and_reserves_one_device(self):
        self.assertNotIn("deploy", self.model["services"]["agentic"])
        model = validation.render_compose(self.env, gpu=True)
        validation.validate_model(model)
        devices = model["services"]["agentic"]["deploy"]["resources"]["reservations"]["devices"]
        self.assertEqual(devices, [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}])

    def test_missing_build_image_fails_instead_of_selecting_latest(self):
        missing = self.root / "missing-image.env"
        missing.write_text(self.env.read_text().replace("PYTHON_RUNTIME_IMAGE=python:fixture@sha256:" + "1" * 64, "PYTHON_RUNTIME_IMAGE="))
        with self.assertRaisesRegex(validation.DeploymentError, "compose_configuration_rejected"):
            validation.render_compose(missing)

    def test_machine_profile_overlay_mounts_reviewed_files_separately_from_credentials(self):
        model = validation.render_compose(self.env, machine_profile=True, gpu=True)
        validation.validate_model(model)
        ai = model["services"]["agentic"]
        mounts = {item["target"]: item for item in ai["volumes"]}
        self.assertTrue(mounts["/run/janus/profiles"]["read_only"])
        self.assertTrue(mounts["/run/secrets/itsm-links.json"]["read_only"])
        self.assertNotEqual(mounts["/run/janus/profiles"]["source"], mounts["/run/secrets/itsm-links.json"]["source"])
        self.assertEqual(ai["environment"]["ITSM_LUA_ADAPTER_FILE"], "/run/janus/profiles/machine.lua")

    def test_machine_profile_cannot_use_writable_missing_or_unpinned_artifacts(self):
        original = validation.render_compose(self.env, machine_profile=True)
        for target in ("/run/janus/profiles", "/run/secrets/itsm-links.json"):
            model = copy.deepcopy(original)
            next(item for item in model["services"]["agentic"]["volumes"] if item["target"] == target)["read_only"] = False
            with self.assertRaisesRegex(validation.DeploymentError, "lua_profile_mounts_must_be_read_only"):
                validation.validate_model(model)
        for value in ("", "placeholder", "a" * 63, "A" * 64):
            model = copy.deepcopy(original)
            model["services"]["agentic"]["environment"]["ITSM_LUA_ADAPTER_SHA256"] = value
            with self.assertRaisesRegex(validation.DeploymentError, "lua_profile_binding_invalid"):
                validation.validate_model(model)
        model = copy.deepcopy(original)
        model["services"]["agentic"]["volumes"] = [item for item in model["services"]["agentic"]["volumes"] if item["target"] != "/run/secrets/itsm-links.json"]
        with self.assertRaisesRegex(validation.DeploymentError, "ai_mount_layout_mismatch"):
            validation.validate_model(model)

    def test_machine_profile_requires_its_explicit_mounts(self):
        self.reject(lambda s: s["agentic"]["environment"].update(ITSM_LUA_ADAPTER_FILE="/workspace/unreviewed.lua"), "lua_profile_mounts_missing")

    def test_private_ca_overlay_scopes_trust_without_disabling_verification(self):
        model = validation.render_compose(self.env, machine_profile=True, private_ca=True)
        validation.validate_model(model)
        for name, variable in (("agentic", "ITSM_HTTP_CA_BUNDLE"), ("frontend", "CURL_CA_BUNDLE")):
            service = model["services"][name]
            target = service["environment"][variable]
            self.assertEqual(target, "/run/janus/trust/ca-bundle.pem")
            self.assertTrue(next(item for item in service["volumes"] if item["target"] == target)["read_only"])

    def test_private_ca_overlay_requires_matching_readonly_trust_mounts(self):
        original = validation.render_compose(self.env, private_ca=True)
        for name in ("agentic", "frontend"):
            model = copy.deepcopy(original)
            next(item for item in model["services"][name]["volumes"] if item["target"] == "/run/janus/trust/ca-bundle.pem")["read_only"] = False
            with self.assertRaisesRegex(validation.DeploymentError, "tls_trust_binding_invalid"):
                validation.validate_model(model)
        model = copy.deepcopy(original)
        model["services"]["agentic"]["environment"]["ITSM_HTTP_CA_BUNDLE"] = "/other/file.pem"
        with self.assertRaisesRegex(validation.DeploymentError, "tls_trust_binding_invalid"):
            validation.validate_model(model)
        self.reject(lambda s: s["agentic"]["environment"].update(ITSM_HTTP_CA_BUNDLE="/unmounted-ca.pem"), "tls_trust_mounts_missing")

    def test_internal_ports_privilege_and_docker_socket_are_refused(self):
        self.reject(lambda s: s["agentic"].update(ports=[{"target": 8000, "published": "8000"}]), "internal_port_published")
        self.reject(lambda s: s["backend"].update(privileged=True), "privileged_container_forbidden")
        self.reject(lambda s: s["agentic"]["volumes"].append({"source": "/var/run/docker.sock", "target": "/var/run/docker.sock"}), "docker_socket_forbidden")

    def test_application_containers_cannot_write_their_image(self):
        for name in ("backend", "frontend", "agentic", "schema"):
            self.assertTrue(self.model["services"][name]["read_only"], name)
            self.reject(lambda s, name=name: s[name].pop("read_only"), "application_root_filesystem_must_be_read_only")

    def test_secure_origin_and_callback_bindings_cannot_drift(self):
        self.reject(lambda s: s["backend"]["environment"].update(AUTH_COOKIE_SECURE="false"), "production_endpoint_policy_mismatch")
        self.reject(lambda s: s["backend"]["environment"].update(CORS_ALLOWED_ORIGINS="https://other.example.test"), "public_origin_binding_mismatch")
        self.reject(lambda s: s["agentic"]["environment"].update(PHOENIX_BASE_URL="http://backend:4000"), "private_tls_callback_required")
        self.reject(lambda s: s["frontend"]["ports"][0].update(published="443"), "public_origin_port_mismatch")

    def test_backend_reaches_the_agentic_service_by_name_without_waiting_for_it(self):
        self.assertEqual(self.model["services"]["backend"]["environment"]["AI_SERVICE_URL"], "http://agentic:8000")
        self.reject(lambda s: s["backend"]["environment"].update(AI_SERVICE_URL="http://ai:8000"), "internal_service_urls_mismatch")
        self.reject(lambda s: s["backend"]["depends_on"].update(agentic={"condition": "service_started"}), "startup_dependency_cycle")

    def test_only_the_service_network_is_a_trusted_proxy(self):
        """Per-client sign-in limits need the real client address. Phoenix trusts
        X-Forwarded-For only from the private service network, where nginx is."""
        subnet = self.model["networks"]["service"]["ipam"]["config"][0]["subnet"]
        self.assertEqual(self.model["services"]["backend"]["environment"]["TRUSTED_PROXIES"], subnet)
        self.reject(lambda s: s["backend"]["environment"].update(TRUSTED_PROXIES="0.0.0.0/0"), "trusted_proxy_binding_mismatch")
        self.reject(lambda s: s["backend"]["environment"].pop("TRUSTED_PROXIES"), "trusted_proxy_binding_mismatch")

    def test_state_singleton_and_offline_weights_are_required(self):
        self.reject(lambda s: s["agentic"]["environment"].update(APPROVAL_STORE_PATH="/tmp/approvals.sqlite3"), "durable_ai_state_not_persisted")
        self.reject(lambda s: s["agentic"]["environment"].update(APPROVAL_STORE_PATH="/state/../tmp/approvals.sqlite3"), "durable_ai_state_not_persisted")
        self.reject(lambda s: s["agentic"].update(deploy={"replicas": 2}), "single_ai_process_storage_required")
        self.reject(lambda s: s["agentic"]["environment"].update(HF_HUB_OFFLINE="0"), "unrequested_model_downloads_forbidden")

    def test_continual_state_uses_the_persistent_mount_without_starting_learning(self):
        env = self.model["services"]["agentic"]["environment"]
        self.reject(lambda s: s["agentic"]["environment"].pop("CONTINUAL_STATE_DB"), "durable_ai_state_not_persisted")
        dockerfile = (AI / "Dockerfile").read_text()
        for key in ("CONTINUAL_STATE_DB", "LEARNING_TRAJECTORY_PATH", "LEARNING_CORRECTION_PATH", "LEARNING_CONTEXT_PATH", "LEARNING_RUN_ROOT", "MAILBOX_PROVISIONING_LOCK_PATH"):
            self.assertIn(f"{key}={env[key]}", dockerfile)
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            path = Path(directory) / Path(env["CONTINUAL_STATE_DB"]).relative_to("/state")
            code = """import os, sqlite3
from learning.continual.automation.config import ContinualAutomationSettings
from learning.continual.automation.store import ContinualAutomationStore
settings = ContinualAutomationSettings.from_env()
assert not settings.enabled and not settings.auto_train and not settings.auto_promote
assert str(settings.state_db_path) == os.environ['CONTINUAL_STATE_DB']
ContinualAutomationStore(settings.state_db_path).initialize()
with sqlite3.connect(settings.state_db_path) as conn:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert {'events', 'cycles', 'kv'} <= tables
"""
            isolated_env = {key: value for key, value in os.environ.items() if not key.startswith("CONTINUAL_")}
            isolated_env.update(PYTHONPATH=str(AI), PYTHONDONTWRITEBYTECODE="1", CONTINUAL_STATE_DB=str(path))
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=directory, env=isolated_env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(path.is_file())

    def test_backend_release_has_no_distribution_and_uses_writable_temporary_storage(self):
        env = self.model["services"]["backend"]["environment"]
        self.assertEqual(env["RELEASE_DISTRIBUTION"], "none")
        self.assertEqual(env["RELEASE_TMP"], "/tmp/janus-release")
        self.reject(lambda s: s["backend"]["environment"].update(RELEASE_DISTRIBUTION="sname"), "single_host_release_policy_mismatch")
        self.reject(lambda s: s["backend"]["environment"].update(RELEASE_TMP="/app/tmp"), "single_host_release_policy_mismatch")
        dockerfile = (BACKEND / "Dockerfile").read_text()
        self.assertIn("RELEASE_DISTRIBUTION=none", dockerfile)
        self.assertIn("RELEASE_TMP=/tmp/janus-release", dockerfile)

    def test_schema_bootstrap_is_not_an_automatic_service_dependency(self):
        self.reject(lambda s: s["schema"].update(profiles=[]), "schema_bootstrap_must_be_explicit")
        self.reject(lambda s: s["backend"]["depends_on"].update(schema={"condition": "service_completed_successfully"}), "automatic_schema_bootstrap_forbidden")


class DeploymentExampleTests(unittest.TestCase):
    """The operator's template and the Compose files name the same settings."""

    def settings(self):
        lines = (DEPLOY / ".env.example").read_text().splitlines()
        return dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))

    def test_example_names_each_application_image_after_its_service(self):
        images = {key: value for key, value in self.settings().items() if key.endswith("_IMAGE") and value}
        self.assertEqual(images, {"AGENTIC_IMAGE": "agentic:local", "BACKEND_IMAGE": "backend:local",
                                  "FRONTEND_IMAGE": "frontend:local"})

    def test_example_and_compose_files_name_the_same_settings(self):
        compose = "".join(path.read_text() for path in sorted(DEPLOY.glob("compose*.yaml")))
        example = set(self.settings())
        self.assertLessEqual(example, set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", compose)))
        self.assertLessEqual(set(re.findall(r"\$\{([A-Z][A-Z0-9_]*):\?", compose)), example)

    def test_each_private_env_file_has_a_companion_example(self):
        files = {key: value for key, value in self.settings().items() if key.endswith("_ENV_FILE")}
        self.assertEqual(set(files), {"AGENTIC_ENV_FILE", "BACKEND_ENV_FILE", "DATABASE_ENV_FILE"})
        for key, value in files.items():
            with self.subTest(key=key):
                self.assertTrue((DEPLOY / (Path(value).name + ".example")).is_file(), value)


class EntrypointTests(unittest.TestCase):
    def test_linux_startup_helpers_retain_lf_on_windows_checkouts(self):
        for repository in (BACKEND, FRONTEND):
            for name in ("docker/entrypoint.sh", "docker/healthcheck.sh"):
                with self.subTest(repository=repository.name, script=name):
                    result = subprocess.run(["git", "-C", str(repository), "check-attr", "eol", "--", name],
                                            capture_output=True, text=True, check=True)
                    self.assertEqual(result.stdout.strip(), f"{name}: eol: lf")
                    self.assertNotIn(b"\r", (repository / name).read_bytes())

    def test_ai_readiness_requires_the_real_service_receipt(self):
        health = module(AI / "docker/healthcheck.py", "ai_container_health")
        class Response:
            status = 200
            def __init__(self, payload): self.payload = payload
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, _): return self.payload
        for body, expected in ((b'{"status":"ready","service":"itsm-ai"}', 0),
                               (b'{"status":"ok"}', 1), (b'{"status":"ready","service":"other"}', 1),
                               (b'not-json', 1), (b'{"status":"ready","service":"itsm-ai","extra":true}', 1)):
            with self.subTest(body=body), patch.object(health, "urlopen", return_value=Response(body)):
                self.assertEqual(health.main(), expected)
        with patch.object(health, "urlopen", side_effect=OSError("unavailable")):
            self.assertEqual(health.main(), 1)

    def test_ai_entrypoint_import_does_not_start_models_and_fixes_one_worker(self):
        entry = module(AI / "docker/entrypoint.py", "ai_container_entry")
        with patch.object(entry.os, "execvp") as execute:
            entry.main()
        args = execute.call_args.args[1]
        self.assertEqual(args[-2:], ["--workers", "1"])
        self.assertEqual(args[3], "api.app:app")

    @unittest.skipUnless(shutil.which("envsubst"), "envsubst is unavailable; frontend rendering was not checked")
    def test_frontend_renders_an_actual_origin_and_keeps_nginx_socket_variables(self):
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            root = Path(directory)
            runtime = root / "runtime"
            cert = root / "tls.crt"
            key = root / "tls.key"
            cert.write_text("synthetic-readable-certificate-not-a-live-cert")
            key.write_text("synthetic-readable-key-not-a-live-key")
            # The image's web root is read-only; runtime values go to the container's tmpfs.
            env = dict(os.environ, PUBLIC_HOST="itsm.example.test", PUBLIC_ORIGIN="https://itsm.example.test:8443",
                       FRONTEND_RUNTIME_DIR=str(runtime), NGINX_CONFIG_TEMPLATE=str(FRONTEND / "docker/nginx.conf.template"),
                       NGINX_CONFIG_OUTPUT=str(root / "nginx.conf"), TLS_CERT_PATH=str(cert), TLS_KEY_PATH=str(key))
            result = subprocess.run(["sh", str(FRONTEND / "docker/entrypoint.sh"), "true"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = (runtime / "config.js").read_text()
            self.assertIn('backendBaseUrl:"https://itsm.example.test:8443"', config)
            self.assertNotIn("requiredOperations", config)
            rendered = (root / "nginx.conf").read_text()
            public, private = rendered.split("listen 9443 ssl;", 1)
            served = re.search(r"location = /config\.js\s*\{([^}]+)\}", public)
            self.assertIsNotNone(served, "the browser must receive the rendered runtime configuration")
            self.assertIn("root " + str(runtime) + ";", served.group(1))
            # Never cached, so a browser always reads the current origin after a redeploy.
            self.assertIn("add_header Cache-Control 'no-store' always;", served.group(1))
            self.assertIn("$http_upgrade", rendered)
            self.assertIn("$connection_upgrade", rendered)
            self.assertNotIn("${PUBLIC_HOST}", rendered)
            self.assertNotIn("${FRONTEND_RUNTIME_DIR}", rendered)
            self.assertIn("ssl_certificate " + str(cert), rendered)
            api = re.search(r"location /api/\s*\{([^}]+)\}", public)
            assert api is not None, "the browser's API location must exist"
            # nginx is the edge: it replaces any X-Forwarded-For a client sent with the
            # address it saw, so a client cannot choose the address the backend's
            # per-client sign-in limits count.
            self.assertIn("proxy_set_header X-Forwarded-For $remote_addr;", api.group(1))
            self.assertNotIn("$proxy_add_x_forwarded_for", rendered)
            health = re.search(r"location = /api/health\s*\{([^}]+)\}", private)
            self.assertIsNotNone(health, "Operations needs its real backend health endpoint on the callback origin")
            self.assertIn("proxy_pass http://phoenix;", health.group(1))
            self.assertIn("proxy_set_header X-Forwarded-Proto https;", health.group(1))
            self.assertNotRegex(private, r"location\s+(?:\^~\s+)?/api/\s*\{")
            self.assertIn("location / { return 404; }", private)
            # Scripts and styles are requested with a content digest (?v=...), so a
            # browser may keep them; each release changes the address. The page itself
            # and its runtime configuration are always revalidated or never stored.
            assets = re.search(r"location ~\* \\\.\(\?:js\|css\|svg\)\$\s*\{([^}]+)\}", public)
            self.assertIsNotNone(assets, "content-stamped files need their own cache rule")
            self.assertIn("gzip_static on;", assets.group(1))
            self.assertIn("include /opt/janus/security-headers.conf;", assets.group(1))
            self.assertRegex(assets.group(1), r"add_header Cache-Control 'public, max-age=\d+, immutable' always;")
            page = re.search(r"location / \{([^}]+)\}", public)
            assert page is not None, "the page's own location must exist"
            self.assertIn("add_header Cache-Control 'no-cache' always;", page.group(1))
            self.assertIn("include /opt/janus/security-headers.conf;", page.group(1))

    def test_frontend_image_ships_compressed_copies_of_its_scripts_and_styles(self):
        dockerfile = (FRONTEND / "Dockerfile").read_text()
        # Compressed once at build time; nginx serves the .gz file to browsers that accept it.
        self.assertRegex(dockerfile, r"gzip -9 -k -n")
        self.assertIn("-name '*.js' -o -name '*.css' -o -name '*.svg'", dockerfile)

    def test_frontend_rejects_an_unsafe_runtime_directory(self):
        for runtime in ("relative/runtime", "/tmp/runtime;}", "/tmp/run time", "/tmp/runtime\nmalicious"):
            result = subprocess.run(["sh", str(FRONTEND / "docker/entrypoint.sh"), "true"], capture_output=True,
                                    env=dict(os.environ, PUBLIC_HOST="itsm.example.test", PUBLIC_ORIGIN="https://itsm.example.test",
                                             FRONTEND_RUNTIME_DIR=runtime))
            self.assertEqual(result.returncode, 64, runtime)

    def test_frontend_rejects_injected_insecure_or_mismatched_origins(self):
        for origin in ("http://itsm.example.test", 'https://itsm.example.test\";alert(1)//',
                       "https://other.example.test", "https://itsm.example.test/path", "https://itsm.example.test\nmalicious",
                       "https://itsm.example.test:0", "https://itsm.example.test:65536", "https://itsm.example.test:443"):
            result = subprocess.run(["sh", str(FRONTEND / "docker/entrypoint.sh"), "true"],
                                    env=dict(os.environ, PUBLIC_HOST="itsm.example.test", PUBLIC_ORIGIN=origin), capture_output=True)
            self.assertEqual(result.returncode, 64)

    def test_backend_probe_requires_http_200_and_backend_identity(self):
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            root = Path(directory)
            curl = root / "curl"
            curl.write_text('#!/usr/bin/env python3\nimport os,sys\na=sys.argv; open(a[a.index("--output")+1],"w").write(os.environ["FIXTURE_BODY"]); print(os.environ["FIXTURE_STATUS"],end="")\n')
            curl.chmod(0o755)
            for status, body, expected in (("200", '{"status":"ok","service":"itsm_backend"}', 0),
                                           ("301", '{"status":"ok","service":"itsm_backend"}', 1),
                                           ("200", '{"status":"ok","service":"other"}', 1), ("200", '{}', 1)):
                result = subprocess.run(["sh", str(BACKEND / "docker/healthcheck.sh")], capture_output=True,
                                        env=dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"],
                                                 FIXTURE_BODY=body, FIXTURE_STATUS=status))
                self.assertEqual(result.returncode == 0, expected == 0)

    def test_backend_entrypoint_does_not_accept_arbitrary_release_commands(self):
        result = subprocess.run(["sh", str(BACKEND / "docker/entrypoint.sh"), "eval", "untrusted"], capture_output=True)
        self.assertEqual(result.returncode, 64)
        script = (BACKEND / "docker/entrypoint.sh").read_text()
        self.assertIn("Application.ensure_all_started(:req)", script)
        self.assertNotIn("Application.ensure_all_started(:itsm_backend)", script)


class RemoteAccessTests(unittest.TestCase):
    """Reaching the stack from other machines without trusting the network in between.

    The application does not depend on how it is reached (Tailscale or Headscale on
    the host, company port forwarding, a cloud load balancer). It needs a bind
    address, an origin, a certificate and, behind a proxy, the proxy ranges that
    may name the real client for the per-client sign-in limits.
    """

    def render(self, trusted):
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            root = Path(directory)
            cert, key = root / "tls.crt", root / "tls.key"
            cert.write_text("synthetic-readable-certificate-not-a-live-cert")
            key.write_text("synthetic-readable-key-not-a-live-key")
            env = dict(os.environ, PUBLIC_HOST="itsm.example.test", PUBLIC_ORIGIN="https://itsm.example.test:8443",
                       FRONTEND_RUNTIME_DIR=str(root / "runtime"), TLS_CERT_PATH=str(cert), TLS_KEY_PATH=str(key),
                       NGINX_CONFIG_TEMPLATE=str(FRONTEND / "docker/nginx.conf.template"),
                       NGINX_CONFIG_OUTPUT=str(root / "nginx.conf"), EDGE_TRUSTED_PROXIES=trusted)
            result = subprocess.run(["sh", str(FRONTEND / "docker/entrypoint.sh"), "true"], env=env,
                                    capture_output=True, text=True)
            if result.returncode:
                return result.returncode, None, None
            rendered = (root / "nginx.conf").read_text()
            included = (root / "runtime" / "real-ip.conf").read_text()
            return result.returncode, rendered, included

    @unittest.skipUnless(shutil.which("envsubst"), "envsubst is unavailable; frontend rendering was not checked")
    def test_a_trusted_edge_proxy_names_the_real_client(self):
        code, rendered, included = self.render("10.0.0.0/16, 2001:db8::/32,192.0.2.10")
        self.assertEqual(code, 0)
        assert rendered is not None and included is not None
        public = rendered.split("listen 9443 ssl;", 1)[0]
        self.assertRegex(public, r"include \S+/real-ip\.conf;")
        self.assertEqual(included.splitlines(), [
            "set_real_ip_from 10.0.0.0/16;", "set_real_ip_from 2001:db8::/32;", "set_real_ip_from 192.0.2.10;",
            "real_ip_header X-Forwarded-For;", "real_ip_recursive on;"])
        # The backend still receives exactly one address: the client nginx settled on.
        self.assertIn("proxy_set_header X-Forwarded-For $remote_addr;", public)

    @unittest.skipUnless(shutil.which("envsubst"), "envsubst is unavailable; frontend rendering was not checked")
    def test_without_a_trusted_proxy_the_connecting_address_is_the_client(self):
        code, _, included = self.render("")
        self.assertEqual((code, included), (0, ""))

    def test_unsafe_trusted_proxy_lists_are_refused(self):
        for value in ("0.0.0.0/0", "::/0", "10.0.0.0/00", "10.0.0.0/8; return 200", "10.0.0.0/8\nmalicious",
                      "edge.example.test", "*", "10.0.0.0/8/1", "10.0.0.0/x"):
            code, _, _ = self.render(value)
            self.assertEqual(code, 64, value)


class StackCommandTests(unittest.TestCase):
    """One command starts, stops and checks the whole stack in dependency order."""

    @classmethod
    def setUpClass(cls):
        cls.janus = module(DEPLOY / "janus.py", "janus_stack_command")

    def plan(self, action, **options):
        return [" ".join(step) for step in self.janus.plan(action, Path("/fixture/deployment.env"), **options)]

    def test_up_validates_then_starts_every_service_in_order_and_waits(self):
        steps = self.plan("up")
        self.assertIn("deploy/validate.py --env-file /fixture/deployment.env", steps[0])
        compose = steps[1:]
        self.assertTrue(all(step.startswith("docker compose --env-file /fixture/deployment.env -f ") for step in compose))
        self.assertTrue(compose[0].endswith("--wait --wait-timeout 120 database"))
        self.assertIn("--profile bootstrap run --rm", compose[1])
        self.assertTrue(compose[1].endswith(" schema"))
        self.assertTrue(compose[2].endswith("--wait --wait-timeout 180 backend frontend"))
        self.assertTrue(compose[3].endswith("--wait --wait-timeout 600 agentic"))
        for step in compose:
            self.assertIn("--pull never", step)
        for step in (compose[0], compose[2], compose[3]):
            self.assertIn(" up -d --no-build ", step)

    def test_overlays_are_added_to_every_step(self):
        for step in self.plan("up", gpu=True, machine_profile=True, private_ca=True):
            if step.startswith("docker compose"):
                for overlay in ("compose.gpu.yaml", "compose.profile.yaml", "compose.trust.yaml"):
                    self.assertIn(overlay, step)
            else:
                self.assertTrue(step.endswith("--gpu --machine-profile --private-ca"))

    def test_stop_keeps_data_and_status_only_reads(self):
        stop, = self.plan("stop")
        self.assertTrue(stop.endswith(" stop --timeout 60"))
        self.assertNotIn(" down", stop)
        status, = self.plan("status")
        self.assertTrue(status.endswith(" ps --all --format json"))

    def test_build_validates_then_builds_the_three_application_images(self):
        validate, build = self.plan("build")
        self.assertIn("deploy/validate.py", validate)
        self.assertTrue(build.endswith(" build backend frontend agentic"))

    def test_unknown_action_and_missing_env_file_are_refused(self):
        with self.assertRaises(ValueError):
            self.janus.plan("destroy", Path("/fixture/deployment.env"))
        result = subprocess.run([sys.executable, str(DEPLOY / "janus.py"), "up", "--env-file", "/does/not/exist.env",
                                 "--dry-run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("env file not found", result.stderr)

    def test_dry_run_prints_the_steps_without_running_them(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", dir=OUT, delete=False) as handle:
            handle.write("COMPOSE_PROJECT_NAME=janus-dry-run-fixture\n")
        try:
            result = subprocess.run([sys.executable, str(DEPLOY / "janus.py"), "up", "--env-file", handle.name,
                                     "--dry-run"], capture_output=True, text=True, timeout=30)
        finally:
            os.unlink(handle.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertTrue(lines[-1].endswith(" agentic"))


class DockerContextTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("docker"), "Docker is unavailable; context exclusion was not verified")
    def test_docker_itself_excludes_nested_dummy_secrets_from_build_context(self):
        for repository in (AI, BACKEND, FRONTEND):
            with self.subTest(repository=repository.name), tempfile.TemporaryDirectory(dir=OUT) as directory:
                root = Path(directory)
                context, output = root / "context", root / "export"
                context.mkdir()
                shutil.copyfile(repository / ".dockerignore", context / ".dockerignore")
                (context / "Dockerfile").write_text("FROM scratch\nCOPY . /checked\n")
                allowed = ("safe.txt", ".env.example", "nested/backend.env.example")
                denied = (".env", "private.env", "nested/.env", "nested/.env.production", "nested/backend.env",
                          "nested/cert.key", "nested/chain.pem", "nested/keys.txt")
                for name in allowed + denied:
                    path = context / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("dummy-context-fixture-only")
                result = subprocess.run(["docker", "buildx", "build", "--progress=quiet", "--network=none",
                                         "--output", "type=local,dest=" + str(output), str(context)],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, "Docker context probe failed: " + result.stderr[-1000:])
                for name in allowed:
                    self.assertTrue((output / "checked" / name).is_file(), name)
                for name in denied:
                    self.assertFalse((output / "checked" / name).exists(), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
