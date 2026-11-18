"""Which worker runs a job: Proxmox first for what fits, the local GPU in parallel, Kaggle for the rest."""
import pytest

from learning.worker_dispatch import Worker, assign, fits


def worker(name, *, kind, device, ram_gib, gpu_gib=0.0, busy=False, available=True, weekly_hours_left=None):
    return Worker(name=name, kind=kind, device=device, ram_gib=ram_gib, gpu_gib=gpu_gib, busy=busy,
                  available=available, weekly_hours_left=weekly_hours_left)


PROXMOX = worker("janus-trainer", kind="proxmox", device="cpu", ram_gib=3.5)
LAPTOP = worker("local-gpu", kind="local", device="cuda", ram_gib=12, gpu_gib=3.2)
KAGGLE = worker("kaggle", kind="kaggle", device="cuda", ram_gib=29, gpu_gib=15, weekly_hours_left=20)


@pytest.mark.parametrize("weights_gib, stages, expected", [
    (0.5, ["sft"], True),            # a 0.5B specialist trains in fp32 LoRA within 3.5 GiB
    (1.4, ["sft"], False),           # 1.4 GiB of bf16 weights become 2.8 GiB in fp32 plus activations
    (0.5, ["sft", "dpo"], True),     # DPO's reference is the frozen base with the adapter off: no copy
    (0.5, ["ppo"], False),           # sampling several answers per prompt on 4 CPU cores is too slow
])
def test_a_cpu_worker_takes_only_what_fits_its_memory_and_speed(weights_gib, stages, expected):
    assert fits(PROXMOX, weights_gib=weights_gib, stages=stages) is expected


def test_qlora_lets_a_small_gpu_train_a_model_whose_weights_exceed_it():
    assert fits(LAPTOP, weights_gib=6.2, stages=["sft"]) is True       # 3B bf16 -> about 1.7 GiB in 4-bit
    assert fits(LAPTOP, weights_gib=15.0, stages=["sft"]) is False     # 7B bf16 -> about 4.3 GiB in 4-bit


def test_proxmox_is_chosen_first_for_a_job_it_can_run():
    plan = assign([("access", 1.4 / 3, ["sft", "dpo"])], [LAPTOP, PROXMOX, KAGGLE])
    assert plan == {"access": "janus-trainer"}


def test_different_jobs_run_side_by_side_on_different_machines():
    jobs = [("access", 0.47, ["sft"]), ("account", 0.47, ["sft"]), ("hub", 6.2, ["sft", "dpo", "ppo"])]
    plan = assign(jobs, [PROXMOX, LAPTOP, KAGGLE])
    assert plan["access"] == "janus-trainer"
    assert plan["account"] == "local-gpu"          # Proxmox is taken by access; the idle laptop GPU helps
    assert plan["hub"] == "kaggle"                 # the 3B hub's PPO needs more than the laptop's free memory


def test_a_busy_or_unavailable_worker_gets_nothing_and_a_job_may_wait():
    plan = assign([("access", 0.47, ["sft"])], [worker("janus-trainer", kind="proxmox", device="cpu", ram_gib=3.5,
                                                      busy=True)])
    assert plan == {"access": None}


def test_kaggle_without_weekly_hours_left_is_not_used():
    spent = worker("kaggle", kind="kaggle", device="cuda", ram_gib=29, gpu_gib=15, weekly_hours_left=0.5)
    assert assign([("hub", 6.2, ["sft"])], [spent]) == {"hub": None}
