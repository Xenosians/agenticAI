from __future__ import annotations

import importlib.util
import sys

from pathlib import (
    Path,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from config import (
    Settings,
)

from subagents.core.definitions.loader import (
    load_agent_definition,
)


JIRA_CANDIDATE_MODEL_KEY = (
    "jira-func"
)


def _dependency_present(
    module_name: str,
) -> bool:
    return (
        importlib.util.find_spec(
            module_name
        )
        is not None
    )


def main(
) -> int:
    settings = (
        Settings()
    )

    jira_agent = (
        load_agent_definition(
            PROJECT_ROOT
            / "subagents"
            / "agents"
            / "jira-specialist.md"
        )
    )

    print(
        "MODEL FLEET PREFLIGHT"
    )
    print(
        "====================="
    )
    print(
        "production Jira model:",
        jira_agent.model,
    )
    print(
        "candidate Jira model:",
        JIRA_CANDIDATE_MODEL_KEY,
    )
    print(
        "max loaded models:",
        settings.model_max_loaded_models,
    )
    effective_pinned = list(
        dict.fromkeys(
            [
                settings.hub_model_key,
                *settings.model_pinned_keys,
            ]
        )
    )

    print(
        "pinned models:",
        effective_pinned,
    )
    print(
        "configured model keys:",
        sorted(
            settings
            .model_profiles
            .keys()
        ),
    )

    failures: list[str] = []

    try:
        profile = (
            settings
            .require_model_profile(
                JIRA_CANDIDATE_MODEL_KEY
            )
        )

    except Exception as exc:
        print(
            "candidate profile: FAIL",
            repr(exc),
        )
        return 1

    print(
        "candidate backend:",
        profile.backend,
    )
    print(
        "candidate path:",
        profile.model_path,
    )
    print(
        "candidate quantization:",
        profile.quantization,
    )
    print(
        "candidate compute dtype:",
        profile.compute_dtype,
    )
    print(
        "candidate device map:",
        profile.device_map,
    )

    if (
        profile.backend
        != "hf-causal"
    ):
        failures.append(
            "jira-func must use the generic hf-causal backend"
        )

    if (
        profile.quantization
        == "bnb4"
        and not _dependency_present(
            "bitsandbytes"
        )
    ):
        failures.append(
            "bitsandbytes is required for jira-func bnb4"
        )

    if (
        jira_agent.model
        == JIRA_CANDIDATE_MODEL_KEY
    ):
        failures.append(
            "jira-func is already promoted; candidate alignment "
            "must not auto-promote production Jira"
        )

    try:
        import torch

        cuda_available = (
            torch.cuda
            .is_available()
        )

        print(
            "cuda available:",
            cuda_available,
        )

        if cuda_available:
            device_index = (
                torch.cuda
                .current_device()
            )

            free_bytes, total_bytes = (
                torch.cuda
                .mem_get_info(
                    device_index
                )
            )

            print(
                "cuda device:",
                torch.cuda.get_device_name(
                    device_index
                ),
            )
            print(
                "cuda free MiB:",
                round(
                    free_bytes
                    / 1024
                    / 1024,
                    1,
                ),
            )
            print(
                "cuda total MiB:",
                round(
                    total_bytes
                    / 1024
                    / 1024,
                    1,
                ),
            )

    except Exception as exc:
        print(
            "cuda diagnostics unavailable:",
            repr(exc),
        )

    if failures:
        print()
        print(
            "PREFLIGHT: FAIL"
        )

        for failure in failures:
            print(
                "-",
                failure,
            )

        return 1

    print()
    print(
        "PREFLIGHT: PASS"
    )
    print(
        "No model was loaded."
    )
    print(
        "No provider call occurred."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
