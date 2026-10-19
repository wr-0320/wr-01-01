"""Common Defender-facing contract for external validation environments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from gymnasium import spaces


class DefenderAction(IntEnum):
    """Environment-neutral actions used by every Stage 17 adapter."""

    OBSERVE = 0
    TRACK = 1
    COMMIT_A = 2
    COMMIT_B = 3


ACTION_NAMES = {
    DefenderAction.OBSERVE: "observe",
    DefenderAction.TRACK: "track",
    DefenderAction.COMMIT_A: "commit_A",
    DefenderAction.COMMIT_B: "commit_B",
}


@dataclass(frozen=True)
class DefenderObservation:
    """A uniform observation plus a non-privileged analytic goal belief."""

    observation: np.ndarray
    belief: tuple[float, float]
    action_mask: tuple[bool, bool, bool, bool]
    step: int
    cumulative_cost: float
    committed_goal: int | None
    source: str

    def __post_init__(self) -> None:
        vector = np.asarray(self.observation)
        belief = np.asarray(self.belief, dtype=float)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("observation must be a finite one-dimensional vector")
        if belief.shape != (2,) or np.any(belief < 0) or not np.isclose(belief.sum(), 1):
            raise ValueError("belief must be a length-two probability distribution")
        if len(self.action_mask) != len(DefenderAction):
            raise ValueError("action_mask must cover all Defender actions")
        if self.step < 0 or self.cumulative_cost < 0:
            raise ValueError("step and cumulative_cost must be nonnegative")
        if self.committed_goal not in (None, 0, 1):
            raise ValueError("committed_goal must be None, 0, or 1")

    def as_dict(self) -> dict[str, object]:
        return {
            "observation": np.asarray(self.observation).tolist(),
            "belief": list(self.belief),
            "action_mask": list(self.action_mask),
            "step": self.step,
            "cumulative_cost": self.cumulative_cost,
            "committed_goal": self.committed_goal,
            "source": self.source,
        }


@dataclass(frozen=True)
class DefenderTransition:
    """The shared evaluator record requested by the Stage 17 protocol.

    ``cost`` is the incremental action cost.  The returned observation carries
    the cumulative cost.  ``success`` remains ``None`` until an episode ends.
    """

    observation: DefenderObservation
    belief: tuple[float, float]
    action: DefenderAction
    cost: float
    success: bool | None
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.belief != self.observation.belief:
            raise ValueError("transition belief must match the returned observation")
        if self.cost < 0:
            raise ValueError("cost must be nonnegative")
        if self.success is not None and not (self.terminated or self.truncated):
            raise ValueError("success is only defined at episode end")

    def as_dict(self) -> dict[str, object]:
        return {
            "observation": self.observation.as_dict(),
            "belief": list(self.belief),
            "action": ACTION_NAMES[self.action],
            "cost": self.cost,
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": dict(self.info),
        }


@runtime_checkable
class DefenderEvaluationEnv(Protocol):
    """Minimal common interface consumed by the unified evaluator."""

    name: str
    action_space: spaces.Discrete
    observation_size: int

    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> DefenderObservation: ...

    def step(self, action: DefenderAction | int) -> DefenderTransition: ...

    def close(self) -> None: ...


def validate_defender_environment(env: DefenderEvaluationEnv) -> dict[str, object]:
    """Exercise the shared fields without accessing a hidden target."""
    if not isinstance(env, DefenderEvaluationEnv):
        raise TypeError("Environment does not implement DefenderEvaluationEnv")
    initial = env.reset(seed=0, options={"goal": "A"})
    if initial.observation.shape != (env.observation_size,):
        raise AssertionError("Observation size does not match the declared interface")
    transition = env.step(DefenderAction.TRACK)
    required = {"observation", "belief", "action", "cost", "success"}
    if not required <= set(transition.as_dict()):
        raise AssertionError("Transition is missing common evaluation fields")
    env.close()
    return {
        "environment": env.name,
        "observation_size": env.observation_size,
        "actions": [ACTION_NAMES[action] for action in DefenderAction],
        "fields": sorted(required),
        "initial_belief": list(initial.belief),
        "track_cost": transition.cost,
    }
