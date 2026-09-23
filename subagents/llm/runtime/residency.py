from __future__ import annotations

from dataclasses import (
    dataclass,
)

from threading import (
    RLock,
)


@dataclass(
    frozen=True
)
class ModelResidencySnapshot:
    """
    Process-local model residency diagnostics.

    max_loaded_models is the configured upper bound.

    loaded_models contains the model keys currently resident in
    the process-local ModelRegistry.

    pinned_models contains configured model keys that the
    residency policy will not select as eviction victims.
    """

    max_loaded_models: int

    loaded_models: tuple[
        str,
        ...,
    ]

    pinned_models: tuple[
        str,
        ...,
    ]


class ModelResidencyController:
    """
    Pure process-local residency policy.

    This controller does not load or unload model objects itself.
    ModelManager owns those lifecycle operations.

    Policy:

        - a bounded number of models may remain loaded;
        - pinned models are never selected as eviction victims;
        - non-pinned models use least-recently-used eviction;
        - loading an already-resident model never causes eviction;
        - when capacity cannot be made without evicting pinned
          models, the request fails closed with a clear error.

    GPU execution admission remains the responsibility of
    GpuScheduler / InferenceCoordinator. Residency and scheduling
    intentionally stay separate concerns.
    """

    def __init__(
        self,
        *,
        max_loaded_models: int,
        pinned_model_keys: (
            list[str]
            | tuple[str, ...]
            | set[str]
        ) = (),
    ) -> None:
        if (
            not isinstance(
                max_loaded_models,
                int,
            )
            or max_loaded_models < 1
        ):
            raise ValueError(
                "max_loaded_models must be a positive integer."
            )

        self.max_loaded_models = (
            max_loaded_models
        )

        self.pinned_model_keys = {
            key.strip()

            for key
            in pinned_model_keys

            if (
                isinstance(
                    key,
                    str,
                )
                and key.strip()
            )
        }

        self._clock = 0

        self._last_used: dict[
            str,
            int,
        ] = {}

        self._lock = (
            RLock()
        )

    # ========================================================
    # USAGE TRACKING
    # ========================================================

    def touch(
        self,
        model_key: str,
    ) -> None:
        with self._lock:
            self._clock += 1

            self._last_used[
                model_key
            ] = (
                self._clock
            )

    def forget(
        self,
        model_key: str,
    ) -> None:
        with self._lock:
            self._last_used.pop(
                model_key,
                None,
            )

    def clear(
        self,
    ) -> None:
        with self._lock:
            self._last_used.clear()

    # ========================================================
    # EVICTION PLANNING
    # ========================================================

    def plan_load(
        self,
        *,
        target_model_key: str,
        loaded_model_keys: list[str],
    ) -> list[str]:
        """
        Return the exact non-pinned model keys that must be
        unloaded before target_model_key may be loaded.

        No side effects occur here.
        """

        loaded = list(
            dict.fromkeys(
                loaded_model_keys
            )
        )

        if (
            target_model_key
            in loaded
        ):
            return []

        required_evictions = max(
            0,
            (
                len(
                    loaded
                )
                + 1
                - self.max_loaded_models
            ),
        )

        if required_evictions == 0:
            return []

        with self._lock:
            candidates = [
                key

                for key
                in loaded

                if (
                    key
                    not in self.pinned_model_keys
                )
            ]

            candidates.sort(
                key=lambda key: (
                    self._last_used.get(
                        key,
                        0,
                    ),
                    loaded.index(
                        key
                    ),
                    key,
                )
            )

        if (
            len(
                candidates
            )
            < required_evictions
        ):
            pinned_loaded = [
                key

                for key
                in loaded

                if (
                    key
                    in self.pinned_model_keys
                )
            ]

            raise RuntimeError(
                "Model residency capacity cannot be satisfied "
                "without evicting pinned models. "
                f"target={target_model_key!r} "
                f"max_loaded_models={self.max_loaded_models} "
                f"loaded={loaded!r} "
                f"pinned_loaded={pinned_loaded!r}"
            )

        return (
            candidates[
                :required_evictions
            ]
        )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def snapshot(
        self,
        loaded_model_keys: list[str],
    ) -> ModelResidencySnapshot:
        return (
            ModelResidencySnapshot(
                max_loaded_models=(
                    self.max_loaded_models
                ),

                loaded_models=tuple(
                    loaded_model_keys
                ),

                pinned_models=tuple(
                    sorted(
                        self.pinned_model_keys
                    )
                ),
            )
        )
