from pathlib import Path


def test_jira_eval_uses_runtime_scheduler_stack():
    source = (
        Path(
            "scripts/jira_specialist_model_eval.py"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "ModelManager"
        in source
    )

    assert (
        "GpuScheduler"
        in source
    )

    assert (
        "InferenceCoordinator"
        in source
    )

    assert (
        "InferencePriority.SPECIALIST"
        in source
    )

    assert (
        "HFCausalWorkerBackend("
        not in source
    )


def test_jira_eval_defaults_to_profile_driven_bnb4():
    source = (
        Path(
            "scripts/jira_specialist_model_eval.py"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        '"bnb4"'
        in source
    )

    assert (
        '"bfloat16"'
        in source
    )

    assert (
        'device_map=MODEL_DEVICE_MAP'
        in source
    )

    assert (
        "model_manager.unload_all()"
        in source
        or (
            "model_manager\n"
            "            .unload_all()"
        ) in source
    )
