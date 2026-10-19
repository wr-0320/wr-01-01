"""Adversarial multi-source intelligence and commitment environment.

The environment models information presentation rather than combat.  A hidden
operational hypothesis generates activity, sensors turn that activity into
delayed and noisy reports, and the red agent can spend a finite budget to
conceal or fabricate activity.  The blue agent never receives simulator truth
labels; it must decide whether to wait, verify, track, or commit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv


REGION_NAMES = ("north", "central", "south")
SOURCE_NAMES = ("aerial", "signals", "ground")
OBJECTIVE_NAMES = ("mobility_corridor", "command_network", "logistics_hub")
PHASE_NAMES = ("preparation", "positioning", "execution")

NUM_REGIONS = len(REGION_NAMES)
NUM_SOURCES = len(SOURCE_NAMES)

RED_HONEST = 0
RED_CONCEAL = 1
RED_DECOY_START = 2
RED_MULTISOURCE_START = RED_DECOY_START + NUM_REGIONS
NUM_RED_ACTIONS = RED_MULTISOURCE_START + NUM_REGIONS

BLUE_WAIT = 0
BLUE_VERIFY_START = 1
BLUE_TRACK_START = BLUE_VERIFY_START + NUM_SOURCES * NUM_REGIONS
BLUE_COMMIT_START = BLUE_TRACK_START + NUM_REGIONS


@dataclass(frozen=True)
class OperationalHypothesis:
    """A joint intent and information-presentation hypothesis."""

    name: str
    main_region: int
    objective: int
    feint_region: int | None = None


OPERATIONAL_HYPOTHESES = (
    OperationalHypothesis("north_corridor_direct", 0, 0),
    OperationalHypothesis("central_command_direct", 1, 1),
    OperationalHypothesis("south_logistics_direct", 2, 2),
    OperationalHypothesis("north_feint_south_logistics", 2, 2, 0),
    OperationalHypothesis("central_feint_north_command", 0, 1, 1),
    OperationalHypothesis("south_feint_central_corridor", 1, 0, 2),
)
NUM_HYPOTHESES = len(OPERATIONAL_HYPOTHESES)
NUM_BLUE_ACTIONS = BLUE_COMMIT_START + NUM_HYPOTHESES


def encode_verify_action(source: int, region: int) -> int:
    if not 0 <= source < NUM_SOURCES or not 0 <= region < NUM_REGIONS:
        raise ValueError("source and region must identify a known sensor and region")
    return BLUE_VERIFY_START + source * NUM_REGIONS + region


def decode_verify_action(action: int) -> tuple[int, int] | None:
    offset = int(action) - BLUE_VERIFY_START
    if 0 <= offset < NUM_SOURCES * NUM_REGIONS:
        return divmod(offset, NUM_REGIONS)
    return None


def encode_track_action(region: int) -> int:
    if not 0 <= region < NUM_REGIONS:
        raise ValueError("region must identify a known region")
    return BLUE_TRACK_START + region


def decode_track_action(action: int) -> int | None:
    region = int(action) - BLUE_TRACK_START
    return region if 0 <= region < NUM_REGIONS else None


def encode_commit_action(hypothesis: int) -> int:
    if not 0 <= hypothesis < NUM_HYPOTHESES:
        raise ValueError("hypothesis must identify a known operational hypothesis")
    return BLUE_COMMIT_START + hypothesis


def decode_commit_action(action: int) -> int | None:
    hypothesis = int(action) - BLUE_COMMIT_START
    return hypothesis if 0 <= hypothesis < NUM_HYPOTHESES else None


def encode_decoy_action(region: int, *, multisource: bool = False) -> int:
    if not 0 <= region < NUM_REGIONS:
        raise ValueError("region must identify a known region")
    return (RED_MULTISOURCE_START if multisource else RED_DECOY_START) + region


def decode_red_action(action: int) -> tuple[str, int | None]:
    action = int(action)
    if action == RED_HONEST:
        return "honest", None
    if action == RED_CONCEAL:
        return "concealment", None
    if RED_DECOY_START <= action < RED_MULTISOURCE_START:
        return "decoy", action - RED_DECOY_START
    if RED_MULTISOURCE_START <= action < NUM_RED_ACTIONS:
        return "multi_source_deception", action - RED_MULTISOURCE_START
    raise ValueError("invalid red action")


class DeceptiveIntelligenceEnv(ParallelEnv):
    """Two-agent adversarial intelligence collection and stopping game.

    Report features are ``activity, confidence, normalized_age, verified``.
    Missing reports remain zero and are distinguished by ``report_mask``.
    Neither the true hypothesis nor deception actions appear in the blue
    observation.  Truth is revealed only in terminal episode metrics.
    """

    metadata = {
        "name": "deceptive_intelligence_v0",
        "render_modes": ["ansi"],
        "is_parallelizable": True,
    }

    _source_reliability = np.array([0.78, 0.72, 0.68], dtype=np.float32)
    _source_coverage = np.array([0.82, 0.90, 0.70], dtype=np.float32)
    _source_delay = np.array([0, 1, 0], dtype=np.int64)
    _source_deception_susceptibility = np.array([0.70, 0.82, 0.45], dtype=np.float32)
    _objective_sensitivity = np.array(
        [
            [0.95, 0.55, 0.70],
            [0.45, 0.98, 0.58],
            [0.72, 0.52, 0.92],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        max_cycles: int = 10,
        deception_budget: int = 4,
        concealment_budget: int = 2,
        verification_budget: int = 3,
        tracking_budget: int = 2,
        render_mode: str | None = None,
    ) -> None:
        if max_cycles < 3:
            raise ValueError("max_cycles must be at least 3")
        if min(deception_budget, concealment_budget, verification_budget, tracking_budget) < 0:
            raise ValueError("budgets must be nonnegative")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")

        self.max_cycles = int(max_cycles)
        self.initial_deception_budget = int(deception_budget)
        self.initial_concealment_budget = int(concealment_budget)
        self.initial_verification_budget = int(verification_budget)
        self.initial_tracking_budget = int(tracking_budget)
        self.render_mode = render_mode

        self.possible_agents = ["red", "blue"]
        self.agents: list[str] = []
        self._action_spaces = {
            "red": spaces.Discrete(NUM_RED_ACTIONS),
            "blue": spaces.Discrete(NUM_BLUE_ACTIONS),
        }
        self._observation_spaces = {
            "red": spaces.Dict(
                {
                    "hypothesis": spaces.Discrete(NUM_HYPOTHESES),
                    "main_region": spaces.Discrete(NUM_REGIONS),
                    "objective": spaces.Discrete(len(OBJECTIVE_NAMES)),
                    "phase": spaces.Discrete(len(PHASE_NAMES)),
                    "step": spaces.Box(0, max_cycles, shape=(1,), dtype=np.int64),
                    "remaining_deception": spaces.Box(
                        0, deception_budget, shape=(1,), dtype=np.int64
                    ),
                    "remaining_concealment": spaces.Box(
                        0, concealment_budget, shape=(1,), dtype=np.int64
                    ),
                    "last_blue_action": spaces.Discrete(NUM_BLUE_ACTIONS + 1),
                    "action_mask": spaces.MultiBinary(NUM_RED_ACTIONS),
                }
            ),
            "blue": spaces.Dict(
                {
                    "reports": spaces.Box(
                        0.0,
                        1.0,
                        shape=(max_cycles, NUM_SOURCES, NUM_REGIONS, 4),
                        dtype=np.float32,
                    ),
                    "report_mask": spaces.MultiBinary(
                        (max_cycles, NUM_SOURCES, NUM_REGIONS)
                    ),
                    "time_mask": spaces.MultiBinary(max_cycles),
                    "source_reliability": spaces.Box(
                        0.0, 1.0, shape=(NUM_SOURCES,), dtype=np.float32
                    ),
                    "remaining_verification": spaces.Box(
                        0, verification_budget, shape=(1,), dtype=np.int64
                    ),
                    "remaining_tracking": spaces.Box(
                        0, tracking_budget, shape=(1,), dtype=np.int64
                    ),
                    "tracked_region": spaces.MultiBinary(NUM_REGIONS),
                    "current_step": spaces.Box(
                        0, max_cycles, shape=(1,), dtype=np.int64
                    ),
                    "action_mask": spaces.MultiBinary(NUM_BLUE_ACTIONS),
                }
            ),
        }
        self.np_random, _ = seeding.np_random(None)
        self._reports = np.zeros(
            (max_cycles, NUM_SOURCES, NUM_REGIONS, 4), dtype=np.float32
        )
        self._report_mask = np.zeros(
            (max_cycles, NUM_SOURCES, NUM_REGIONS), dtype=np.int8
        )
        self._time_mask = np.zeros(max_cycles, dtype=np.int8)
        self._hypothesis = 0
        self._step_count = 0
        self._deception_budget = deception_budget
        self._concealment_budget = concealment_budget
        self._verification_budget = verification_budget
        self._tracking_budget = tracking_budget
        self._tracked_region: int | None = None
        self._tracking_ttl = 0
        self._last_blue_action = NUM_BLUE_ACTIONS
        self._last_red_action = RED_HONEST
        self._total_blue_cost = 0.0
        self._total_red_cost = 0.0
        self._deception_steps = 0
        self._verification_steps = 0

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self._action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        self.np_random, _ = seeding.np_random(seed)
        options = options or {}
        requested = options.get("hypothesis")
        if requested is None:
            self._hypothesis = int(self.np_random.integers(NUM_HYPOTHESES))
        elif isinstance(requested, str):
            names = [hypothesis.name for hypothesis in OPERATIONAL_HYPOTHESES]
            if requested not in names:
                raise ValueError(f"unknown hypothesis: {requested}")
            self._hypothesis = names.index(requested)
        elif 0 <= int(requested) < NUM_HYPOTHESES:
            self._hypothesis = int(requested)
        else:
            raise ValueError("options['hypothesis'] must be a valid index or name")

        self.agents = self.possible_agents.copy()
        self._step_count = 0
        self._deception_budget = self.initial_deception_budget
        self._concealment_budget = self.initial_concealment_budget
        self._verification_budget = self.initial_verification_budget
        self._tracking_budget = self.initial_tracking_budget
        self._tracked_region = None
        self._tracking_ttl = 0
        self._last_blue_action = NUM_BLUE_ACTIONS
        self._last_red_action = RED_HONEST
        self._total_blue_cost = 0.0
        self._total_red_cost = 0.0
        self._deception_steps = 0
        self._verification_steps = 0
        self._reports.fill(0)
        self._report_mask.fill(0)
        self._time_mask.fill(0)
        observations = self._observations()
        return observations, {agent: {} for agent in self.agents}

    def step(
        self, actions: Mapping[str, int]
    ) -> tuple[dict, dict[str, float], dict[str, bool], dict[str, bool], dict]:
        if not self.agents:
            raise RuntimeError("Call reset() before stepping a finished episode")
        if set(actions) != set(self.agents):
            raise ValueError("Actions must be provided for both active agents")
        red_action = int(actions["red"])
        blue_action = int(actions["blue"])
        if not self._action_spaces["red"].contains(red_action):
            raise ValueError("invalid red action")
        if not self._action_spaces["blue"].contains(blue_action):
            raise ValueError("invalid blue action")

        effective_red_action, red_cost = self._apply_red_budget(red_action)
        blue_cost, verification = self._apply_blue_action(blue_action)
        self._total_red_cost += red_cost
        self._total_blue_cost += blue_cost
        self._generate_reports(effective_red_action, verification)
        self._last_red_action = effective_red_action
        self._last_blue_action = blue_action
        self._step_count += 1

        committed = decode_commit_action(blue_action)
        terminated = committed is not None
        truncated = not terminated and self._step_count >= self.max_cycles
        blue_reward = -blue_cost
        red_reward = -red_cost
        metrics: dict[str, Any] | None = None
        if committed is not None:
            correct = committed == self._hypothesis
            decision_value = 10.0 if correct else -10.0
            urgency_penalty = 0.25 * (self._step_count - 1) if correct else 0.0
            blue_reward += decision_value - urgency_penalty
            red_reward += -decision_value + urgency_penalty
            metrics = self._episode_metrics(committed, correct, timed_out=False)
        elif truncated:
            blue_reward -= 8.0
            red_reward += 8.0
            metrics = self._episode_metrics(None, False, timed_out=True)

        rewards = {"red": float(red_reward), "blue": float(blue_reward)}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        observations = self._observations()
        infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}
        if metrics is not None:
            infos = {agent: {"episode_metrics": metrics.copy()} for agent in self.agents}
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        return None

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        lines = [f"step={self._step_count} phase={PHASE_NAMES[self._phase()]}"]
        for source, source_name in enumerate(SOURCE_NAMES):
            visible = []
            if self._step_count:
                index = self._step_count - 1
                for region, region_name in enumerate(REGION_NAMES):
                    if self._report_mask[index, source, region]:
                        value = self._reports[index, source, region, 0]
                        confidence = self._reports[index, source, region, 1]
                        visible.append(f"{region_name}:{value:.2f}@{confidence:.2f}")
            lines.append(f"{source_name}: " + (", ".join(visible) or "no report"))
        return "\n".join(lines)

    def _phase(self, step: int | None = None) -> int:
        step = self._step_count if step is None else step
        fraction = step / max(1, self.max_cycles - 1)
        return min(2, int(fraction * 3))

    def _red_action_mask(self) -> np.ndarray:
        mask = np.ones(NUM_RED_ACTIONS, dtype=np.int8)
        if self._concealment_budget <= 0:
            mask[RED_CONCEAL] = 0
        if self._deception_budget <= 0:
            mask[RED_DECOY_START:] = 0
        elif self._deception_budget < 2:
            mask[RED_MULTISOURCE_START:] = 0
        return mask

    def _blue_action_mask(self) -> np.ndarray:
        mask = np.ones(NUM_BLUE_ACTIONS, dtype=np.int8)
        if self._verification_budget <= 0:
            mask[BLUE_VERIFY_START:BLUE_TRACK_START] = 0
        if self._tracking_budget <= 0:
            mask[BLUE_TRACK_START:BLUE_COMMIT_START] = 0
        return mask

    def _observations(self) -> dict[str, dict[str, Any]]:
        hypothesis = OPERATIONAL_HYPOTHESES[self._hypothesis]
        tracked = np.zeros(NUM_REGIONS, dtype=np.int8)
        if self._tracked_region is not None and self._tracking_ttl > 0:
            tracked[self._tracked_region] = 1
        red = {
            "hypothesis": self._hypothesis,
            "main_region": hypothesis.main_region,
            "objective": hypothesis.objective,
            "phase": self._phase(),
            "step": np.array([self._step_count], dtype=np.int64),
            "remaining_deception": np.array([self._deception_budget], dtype=np.int64),
            "remaining_concealment": np.array([self._concealment_budget], dtype=np.int64),
            "last_blue_action": self._last_blue_action,
            "action_mask": self._red_action_mask(),
        }
        blue = {
            "reports": self._reports.copy(),
            "report_mask": self._report_mask.copy(),
            "time_mask": self._time_mask.copy(),
            "source_reliability": self._source_reliability.copy(),
            "remaining_verification": np.array(
                [self._verification_budget], dtype=np.int64
            ),
            "remaining_tracking": np.array([self._tracking_budget], dtype=np.int64),
            "tracked_region": tracked,
            "current_step": np.array([self._step_count], dtype=np.int64),
            "action_mask": self._blue_action_mask(),
        }
        return {"red": red, "blue": blue}

    def _apply_red_budget(self, action: int) -> tuple[int, float]:
        kind, _ = decode_red_action(action)
        if kind == "concealment":
            if self._concealment_budget <= 0:
                return RED_HONEST, 0.0
            self._concealment_budget -= 1
            self._deception_steps += 1
            return action, 0.20
        if kind == "decoy":
            if self._deception_budget <= 0:
                return RED_HONEST, 0.0
            self._deception_budget -= 1
            self._deception_steps += 1
            return action, 0.25
        if kind == "multi_source_deception":
            if self._deception_budget < 2:
                return RED_HONEST, 0.0
            self._deception_budget -= 2
            self._deception_steps += 1
            return action, 0.50
        return RED_HONEST, 0.0

    def _apply_blue_action(
        self, action: int
    ) -> tuple[float, tuple[int, int] | None]:
        verification = decode_verify_action(action)
        tracked_region = decode_track_action(action)
        if verification is not None:
            if self._verification_budget <= 0:
                return 0.15, None
            self._verification_budget -= 1
            self._verification_steps += 1
            return 0.45, verification
        if tracked_region is not None:
            if self._tracking_budget <= 0:
                return 0.15, None
            self._tracking_budget -= 1
            self._tracked_region = tracked_region
            self._tracking_ttl = 3
            return 0.30, None
        if decode_commit_action(action) is not None:
            return 0.0, None
        return 0.15, None

    def _base_activity(self, source: int, region: int, observed_step: int) -> float:
        hypothesis = OPERATIONAL_HYPOTHESES[self._hypothesis]
        phase_gain = (0.45, 0.68, 0.92)[self._phase(observed_step)]
        sensitivity = float(self._objective_sensitivity[source, hypothesis.objective])
        background = 0.10 + 0.03 * ((source + region) % 2)
        if region == hypothesis.main_region:
            return background + 0.72 * phase_gain * sensitivity
        return background + 0.08 * phase_gain * sensitivity

    def _generate_reports(
        self, red_action: int, verification: tuple[int, int] | None
    ) -> None:
        row = self._step_count
        if row >= self.max_cycles:
            return
        self._time_mask[row] = 1
        kind, red_region = decode_red_action(red_action)
        hypothesis = OPERATIONAL_HYPOTHESES[self._hypothesis]
        for source in range(NUM_SOURCES):
            observed_step = max(0, self._step_count - int(self._source_delay[source]))
            for region in range(NUM_REGIONS):
                verified = verification == (source, region)
                tracked = self._tracked_region == region and self._tracking_ttl > 0
                coverage = float(self._source_coverage[source])
                if verified:
                    coverage = 1.0
                elif tracked:
                    coverage = min(1.0, coverage + 0.16)
                if self.np_random.random() > coverage:
                    continue

                activity = self._base_activity(source, region, observed_step)
                susceptibility = float(self._source_deception_susceptibility[source])
                if verified:
                    susceptibility *= 0.30
                elif tracked:
                    susceptibility *= 0.65
                if kind == "concealment" and region == hypothesis.main_region:
                    activity -= 0.42 * susceptibility
                elif kind == "decoy" and region == red_region:
                    selected_source = (self._step_count + int(red_region)) % NUM_SOURCES
                    if source == selected_source:
                        activity += 0.58 * susceptibility
                elif kind == "multi_source_deception" and region == red_region:
                    activity += 0.46 * susceptibility

                reliability = float(self._source_reliability[source])
                noise_scale = 0.18 * (1.0 - reliability)
                if verified:
                    noise_scale *= 0.35
                elif tracked:
                    noise_scale *= 0.65
                activity += float(self.np_random.normal(0.0, noise_scale))
                activity = float(np.clip(activity, 0.0, 1.0))

                confidence = 0.40 + 0.45 * reliability
                confidence += 0.10 if verified else 0.04 if tracked else 0.0
                confidence += float(self.np_random.normal(0.0, 0.06))
                confidence = float(np.clip(confidence, 0.05, 0.98))
                age = min(
                    1.0,
                    (int(self._source_delay[source]) + (0 if verified else 1))
                    / self.max_cycles,
                )
                self._reports[row, source, region] = (
                    activity,
                    confidence,
                    age,
                    float(verified),
                )
                self._report_mask[row, source, region] = 1
        if self._tracking_ttl > 0:
            self._tracking_ttl -= 1
            if self._tracking_ttl == 0:
                self._tracked_region = None

    def _episode_metrics(
        self, committed: int | None, correct: bool, *, timed_out: bool
    ) -> dict[str, Any]:
        hypothesis = OPERATIONAL_HYPOTHESES[self._hypothesis]
        achieved_utility = (
            -8.0
            if timed_out
            else (10.0 - 0.25 * (self._step_count - 1) if correct else -10.0)
        ) - self._total_blue_cost
        return {
            "true_hypothesis": self._hypothesis,
            "true_hypothesis_name": hypothesis.name,
            "true_main_region": REGION_NAMES[hypothesis.main_region],
            "true_objective": OBJECTIVE_NAMES[hypothesis.objective],
            "planned_feint_region": (
                "" if hypothesis.feint_region is None else REGION_NAMES[hypothesis.feint_region]
            ),
            "committed_hypothesis": -1 if committed is None else committed,
            "decision_correct": int(correct),
            "decision_step": self._step_count,
            "timed_out": int(timed_out),
            "deception_steps": self._deception_steps,
            "verification_steps": self._verification_steps,
            "blue_information_cost": round(self._total_blue_cost, 6),
            "red_deception_cost": round(self._total_red_cost, 6),
            "decision_utility": round(achieved_utility, 6),
            "decision_regret": round(10.0 - achieved_utility, 6),
            "deception_succeeded": int(self._deception_steps > 0 and not correct),
        }

