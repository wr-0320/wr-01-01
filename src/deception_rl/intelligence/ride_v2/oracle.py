"""Privileged simulator diagnostics for RIDe-v2 headroom measurement.

Nothing in this module is a deployable Defender.  Oracle context is supplied by
the offline evaluation harness and never inserted into a blue observation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from deception_rl.envs.deceptive_intelligence import (
    BLUE_WAIT,
    NUM_REGIONS,
    NUM_SOURCES,
    OPERATIONAL_HYPOTHESES,
    RED_HONEST,
    DeceptiveIntelligenceEnv,
    decode_red_action,
    encode_commit_action,
    encode_verify_action,
)
from deception_rl.intelligence.ride import RIDeIntelligenceController
from deception_rl.intelligence.ride_v2.active import (
    BudgetMatchedActiveController,
    MultiSourceEntropyVerify,
)


@dataclass(frozen=True)
class OracleDiagnosticStep:
    """Privileged description of one generated report row."""

    effective_red_action: int
    regime: tuple[str, int | None]
    regime_changed: bool
    effective_source_reliability: np.ndarray
    compromised_cells: np.ndarray
    privileged_diagnostic_only: bool = True


class OracleSourceReliability:
    """Read the report-generation mechanism for diagnostic headroom only."""

    privileged_diagnostic_only = True

    @staticmethod
    def _effective_action(env: DeceptiveIntelligenceEnv, red_action: int) -> int:
        mask = env._red_action_mask()  # noqa: SLF001 - explicit oracle access
        return int(red_action) if mask[int(red_action)] else RED_HONEST

    def inspect(
        self,
        env: DeceptiveIntelligenceEnv,
        red_action: int,
        previous: OracleDiagnosticStep | None,
    ) -> OracleDiagnosticStep:
        action = self._effective_action(env, red_action)
        kind, region = decode_red_action(action)
        compromised = np.zeros((NUM_SOURCES, NUM_REGIONS), dtype=bool)
        reliability = env._source_reliability.astype(float).copy()  # noqa: SLF001
        susceptibility = env._source_deception_susceptibility  # noqa: SLF001
        if kind == "concealment":
            region = OPERATIONAL_HYPOTHESES[env._hypothesis].main_region  # noqa: SLF001
            compromised[:, region] = True
            reliability *= 1.0 - 0.42 * susceptibility
        elif kind == "decoy" and region is not None:
            source = (env._step_count + int(region)) % NUM_SOURCES  # noqa: SLF001
            compromised[source, region] = True
            reliability[source] *= 1.0 - 0.58 * susceptibility[source]
        elif kind == "multi_source_deception" and region is not None:
            compromised[:, region] = True
            reliability *= 1.0 - 0.46 * susceptibility
        regime = (kind, region)
        changed = previous is not None and previous.regime != regime
        return OracleDiagnosticStep(
            effective_red_action=action,
            regime=regime,
            regime_changed=changed,
            effective_source_reliability=np.clip(reliability, 0.0, 1.0),
            compromised_cells=compromised,
        )

    @staticmethod
    def filter_compromised_reports(
        observation: dict[str, Any], trace: Iterable[OracleDiagnosticStep]
    ) -> dict[str, Any]:
        """Return a copy with oracle-identified manipulated cells excluded."""

        filtered = deepcopy(observation)
        mask = np.asarray(filtered["report_mask"]).copy()
        reports = np.asarray(filtered["reports"]).copy()
        steps = list(trace)
        for row, diagnostic in enumerate(steps[: len(mask)]):
            mask[row][diagnostic.compromised_cells] = 0
            reports[row][diagnostic.compromised_cells] = 0
        filtered["report_mask"] = mask
        filtered["reports"] = reports
        if steps:
            filtered["source_reliability"] = steps[-1].effective_source_reliability.astype(
                np.float32
            )
        return filtered


class PrivilegedOracleController:
    privileged_diagnostic_only = True

    def act(self, observation: dict[str, Any]) -> int:
        del observation
        raise RuntimeError("oracle controllers require the isolated diagnostic harness")


class OracleReliabilityController(PrivilegedOracleController):
    """Measure perfect compromised-cell identification at a fixed decision time."""

    name = "oracle_reliability"

    def __init__(
        self, inference: RIDeIntelligenceController, *, commit_step: int = 4
    ) -> None:
        self.inference = inference
        self.commit_step = int(commit_step)
        self.reliability = OracleSourceReliability()
        self.reset()

    def reset(self) -> None:
        self.inference.reset()
        self.info: dict[str, Any] | None = None

    def act_with_oracle(
        self,
        observation: dict[str, Any],
        env: DeceptiveIntelligenceEnv,
        red_action: int,
        trace: list[OracleDiagnosticStep],
    ) -> int:
        del env, red_action
        filtered = self.reliability.filter_compromised_reports(observation, trace)
        self.info = self.inference.infer(filtered)
        self.info["privileged_diagnostic_only"] = True
        self.info["oracle_reliability_rows"] = len(trace)
        step = int(np.asarray(observation["current_step"])[0])
        if step >= self.commit_step:
            return encode_commit_action(int(np.argmax(self.info["hypothesis_belief"])))
        return BLUE_WAIT


class OracleVerifyPolicy(PrivilegedOracleController, BudgetMatchedActiveController):
    """Rank VERIFY(source, region) actions by simulator one-step decision risk."""

    name = "oracle_voi"

    def __init__(
        self,
        inference: RIDeIntelligenceController,
        *,
        seed: int,
        rollouts: int = 6,
        verification_budget: int = 3,
        commit_step: int = 4,
        lambda_cost: float = 0.10,
        lambda_delay: float = 0.02,
    ) -> None:
        if rollouts < 1:
            raise ValueError("rollouts must be positive")
        self.seed = int(seed)
        self.rollouts = int(rollouts)
        self.lambda_cost = float(lambda_cost)
        self.lambda_delay = float(lambda_delay)
        BudgetMatchedActiveController.__init__(
            self,
            inference,
            verification_budget=verification_budget,
            commit_step=commit_step,
        )

    def reset(self) -> None:
        BudgetMatchedActiveController.reset(self)
        self._oracle_call = 0

    def _prepare_observation(
        self,
        observation: dict[str, Any],
        trace: list[OracleDiagnosticStep],
    ) -> dict[str, Any]:
        del trace
        return observation

    def _risk(self, observation: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        info = self.inference.infer(observation)
        belief = np.asarray(info["hypothesis_belief"], dtype=float)
        return 1.0 - float(np.max(belief)), info

    def _candidate_value(
        self,
        env: DeceptiveIntelligenceEnv,
        red_action: int,
        action: int,
        trace: list[OracleDiagnosticStep],
        risk_before: float,
    ) -> float:
        next_diagnostic = OracleSourceReliability().inspect(
            env, red_action, trace[-1] if trace else None
        )
        risks = []
        step = int(env._step_count)  # noqa: SLF001 - explicit oracle access
        for rollout in range(self.rollouts):
            simulated = deepcopy(env)
            simulated.np_random = np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, self._oracle_call, step, action, rollout]
                )
            )
            observations, *_ = simulated.step(
                {"red": red_action, "blue": action}
            )
            prepared = self._prepare_observation(
                observations["blue"], trace + [next_diagnostic]
            )
            risk, _ = self._risk(prepared)
            risks.append(risk)
        return (
            risk_before
            - float(np.mean(risks))
            - self.lambda_cost * 0.45
            - self.lambda_delay
        )

    def act_with_oracle(
        self,
        observation: dict[str, Any],
        env: DeceptiveIntelligenceEnv,
        red_action: int,
        trace: list[OracleDiagnosticStep],
    ) -> int:
        prepared = self._prepare_observation(observation, trace)
        risk_before, self.info = self._risk(prepared)
        self.info["privileged_diagnostic_only"] = True
        predicted = int(np.argmax(self.info["hypothesis_belief"]))
        step = int(np.asarray(observation["current_step"])[0])
        remaining = int(np.asarray(observation["remaining_verification"])[0])
        if step >= self.commit_step:
            return encode_commit_action(predicted)
        if len(self._verified_cells) < self.verification_budget and remaining > 0:
            candidates = self._legal_untried_cells(observation)
            values = {
                cell: self._candidate_value(
                    env,
                    red_action,
                    encode_verify_action(*cell),
                    trace,
                    risk_before,
                )
                for cell in candidates
            }
            cell = max(values, key=values.get)
            self._verified_cells.append(cell)
            self.info["selected_verification"] = list(cell)
            self.info["oracle_voi"] = {
                f"{source}:{region}": value
                for (source, region), value in values.items()
            }
            self._oracle_call += 1
            return encode_verify_action(*cell)
        return BLUE_WAIT


class OracleChangePoint(PrivilegedOracleController):
    """Reset old evidence and request a verification after a true regime switch."""

    name = "oracle_change"

    def __init__(
        self, inference: RIDeIntelligenceController, *, commit_step: int = 4
    ) -> None:
        self.inference = inference
        self.commit_step = int(commit_step)
        self.selector = MultiSourceEntropyVerify(
            inference, verification_budget=3, commit_step=commit_step
        )
        self.reset()

    def reset(self) -> None:
        self.inference.reset()
        self.selector.reset()
        self.info: dict[str, Any] | None = None
        self._change_cooldown = 0

    @staticmethod
    def _latest_only(observation: dict[str, Any]) -> dict[str, Any]:
        current = deepcopy(observation)
        active = np.flatnonzero(np.asarray(current["time_mask"], dtype=bool))
        if len(active):
            time_mask = np.zeros_like(current["time_mask"])
            time_mask[int(active[-1])] = 1
            current["time_mask"] = time_mask
        return current

    def act_with_oracle(
        self,
        observation: dict[str, Any],
        env: DeceptiveIntelligenceEnv,
        red_action: int,
        trace: list[OracleDiagnosticStep],
    ) -> int:
        del env, red_action
        changed = bool(trace and trace[-1].regime_changed)
        if changed:
            self._change_cooldown = 1
        prepared = self._latest_only(observation) if self._change_cooldown else observation
        self.info = self.inference.infer(prepared)
        self.info["privileged_diagnostic_only"] = True
        self.info["oracle_change_detected"] = changed
        step = int(np.asarray(observation["current_step"])[0])
        if self._change_cooldown:
            self._change_cooldown -= 1
            remaining = int(np.asarray(observation["remaining_verification"])[0])
            if remaining > 0:
                belief = np.asarray(self.info["hypothesis_belief"], dtype=float)
                cell = self.selector._verification_cell(observation, belief)
                self.selector._verified_cells.append(cell)
                return encode_verify_action(*cell)
            return BLUE_WAIT
        if step >= self.commit_step:
            return encode_commit_action(int(np.argmax(self.info["hypothesis_belief"])))
        return BLUE_WAIT


class FullOracleDefender(OracleVerifyPolicy):
    """Combine oracle reliability filtering, VOI, and change-aware decay."""

    name = "full_oracle"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reliability = OracleSourceReliability()

    def _prepare_observation(
        self,
        observation: dict[str, Any],
        trace: list[OracleDiagnosticStep],
    ) -> dict[str, Any]:
        filtered = self.reliability.filter_compromised_reports(observation, trace)
        if trace and trace[-1].regime_changed:
            filtered = OracleChangePoint._latest_only(filtered)
        return filtered
