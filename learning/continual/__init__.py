from learning.continual.checkpoints import AdapterCheckpointStore
from learning.continual.controller import ContinualLearningController, run_continual_cycle
from learning.continual.cycle import (
    DEFAULT_CONTEXT_EVENTS,
    DEFAULT_CONTEXT_RECORDS,
    DEFAULT_CONTINUAL_ROOT,
    ContinualLearningCycle,
    ContinualLearningPolicy,
)
from learning.continual.materialization import TrainingMaterializationPlan, build_materialization_plan
from learning.continual.replay import ReplayPolicy, select_replay
from learning.continual.triggers import TrainingTriggerDecision, TrainingTriggerPolicy, evaluate_training_trigger
from learning.continual.types import (
    ActiveCheckpointPointer,
    AdapterCheckpointManifest,
    ContinualCycleManifest,
    ReplaySelection,
    WeakLearningSignal,
)
from learning.continual.weak_supervision import classify_trajectories, classify_trajectory
from learning.continual.windows import EvidenceWindow, build_evidence_window

__all__ = [
    "ActiveCheckpointPointer",
    "AdapterCheckpointManifest",
    "AdapterCheckpointStore",
    "ContinualCycleManifest",
    "ContinualLearningController",
    "ContinualLearningCycle",
    "ContinualLearningPolicy",
    "DEFAULT_CONTEXT_EVENTS",
    "DEFAULT_CONTEXT_RECORDS",
    "DEFAULT_CONTINUAL_ROOT",
    "EvidenceWindow",
    "ReplayPolicy",
    "ReplaySelection",
    "TrainingMaterializationPlan",
    "TrainingTriggerDecision",
    "TrainingTriggerPolicy",
    "WeakLearningSignal",
    "build_evidence_window",
    "build_materialization_plan",
    "classify_trajectories",
    "classify_trajectory",
    "evaluate_training_trigger",
    "run_continual_cycle",
    "select_replay",
]
