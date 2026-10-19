"""Budget-matched active verification baselines using blue-visible inputs only."""

from __future__ import annotations

from typing import Any

import numpy as np

from deception_rl.envs.deceptive_intelligence import (
    BLUE_WAIT,
    NUM_HYPOTHESES,
    NUM_REGIONS,
    NUM_SOURCES,
    encode_commit_action,
    encode_verify_action,
)
from deception_rl.intelligence.ride import RIDeIntelligenceController


class BudgetMatchedActiveController:
    """Verify under a fixed budget, then commit at a shared decision time."""

    privileged_diagnostic_only = False

    def __init__(
        self,
        inference: RIDeIntelligenceController,
        *,
        verification_budget: int = 3,
        commit_step: int = 4,
    ) -> None:
        if verification_budget < 1:
            raise ValueError("verification_budget must be positive")
        if commit_step < verification_budget:
            raise ValueError("commit_step must allow the shared verification budget")
        self.inference = inference
        self.verification_budget = int(verification_budget)
        self.commit_step = int(commit_step)
        self.reset()

    def reset(self) -> None:
        self.inference.reset()
        self.info: dict[str, Any] | None = None
        self._verified_cells: list[tuple[int, int]] = []

    def _infer(self, observation: dict[str, Any]) -> np.ndarray:
        self.info = self.inference.infer(observation)
        return np.asarray(self.info["hypothesis_belief"], dtype=float)

    def _verification_cell(
        self, observation: dict[str, Any], belief: np.ndarray
    ) -> tuple[int, int]:
        raise NotImplementedError

    def _legal_untried_cells(self, observation: dict[str, Any]) -> list[tuple[int, int]]:
        mask = np.asarray(observation["action_mask"], dtype=bool)
        cells = [
            (source, region)
            for source in range(NUM_SOURCES)
            for region in range(NUM_REGIONS)
            if mask[encode_verify_action(source, region)]
        ]
        untried = [cell for cell in cells if cell not in self._verified_cells]
        return untried or cells

    def act(self, observation: dict[str, Any]) -> int:
        belief = self._infer(observation)
        predicted = int(np.argmax(belief))
        step = int(np.asarray(observation["current_step"])[0])
        remaining = int(np.asarray(observation["remaining_verification"])[0])
        if step >= self.commit_step:
            return encode_commit_action(predicted)
        if len(self._verified_cells) < self.verification_budget and remaining > 0:
            cell = self._verification_cell(observation, belief)
            self._verified_cells.append(cell)
            self.info["selected_verification"] = list(cell)
            self.info["verification_selection"] = self.name
            return encode_verify_action(*cell)
        return BLUE_WAIT


class MultiSourceRandomVerify(BudgetMatchedActiveController):
    """Choose verification cells uniformly without replacement."""

    name = "multi_source_random_verify"

    def __init__(
        self,
        inference: RIDeIntelligenceController,
        *,
        seed: int,
        verification_budget: int = 3,
        commit_step: int = 4,
    ) -> None:
        self.seed = int(seed)
        super().__init__(
            inference,
            verification_budget=verification_budget,
            commit_step=commit_step,
        )

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.seed)

    def _verification_cell(
        self, observation: dict[str, Any], belief: np.ndarray
    ) -> tuple[int, int]:
        del belief
        candidates = self._legal_untried_cells(observation)
        return candidates[int(self._rng.integers(len(candidates)))]


class MultiSourceEntropyVerify(BudgetMatchedActiveController):
    """Verify the cell with greatest belief-weighted hypothesis disagreement."""

    name = "multi_source_entropy_verify"

    def _verification_cell(
        self, observation: dict[str, Any], belief: np.ndarray
    ) -> tuple[int, int]:
        if belief.shape != (NUM_HYPOTHESES,):
            raise ValueError("belief has an unexpected shape")
        templates = self.inference.reasoner.prototypes
        activity_indices = np.arange(NUM_SOURCES * NUM_REGIONS) * 5
        activity = templates[:, activity_indices].reshape(
            NUM_HYPOTHESES, NUM_SOURCES, NUM_REGIONS
        )
        mean = np.sum(belief[:, None, None] * activity, axis=0)
        disagreement = np.sum(
            belief[:, None, None] * (activity - mean[None, ...]) ** 2,
            axis=0,
        )
        candidates = self._legal_untried_cells(observation)
        return max(candidates, key=lambda cell: float(disagreement[cell]))
