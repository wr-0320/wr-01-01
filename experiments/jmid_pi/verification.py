"""Exact arithmetic witnesses for JMID-PI; this is not a prediction algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping


F = Fraction
Outcome = str


@dataclass(frozen=True)
class LatentConstruction:
    """Binary-state construction with three perfectly duplicated reports.

    ``raw_positive[state]`` is P(R=1|Z=state). ``observe_given_raw[state]``
    contains P(M=1|Z=state,R=0) and P(M=1|Z=state,R=1). All three sources
    share R and M, so the only observed outcomes are ***, 000, and 111.
    """

    name: str
    graph: tuple[tuple[int, int], ...]
    prior: Fraction
    raw_positive: tuple[Fraction, Fraction]
    observe_given_raw: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]

    def conditional_observed(self, state: int) -> dict[Outcome, Fraction]:
        theta = self.raw_positive[state]
        observe_zero, observe_one = self.observe_given_raw[state]
        visible_one = theta * observe_one
        visible_zero = (1 - theta) * observe_zero
        return {"***": 1 - visible_zero - visible_one, "000": visible_zero, "111": visible_one}

    def observed_marginal(self) -> dict[Outcome, Fraction]:
        q0, q1 = self.conditional_observed(0), self.conditional_observed(1)
        return {outcome: (1 - self.prior) * q0[outcome] + self.prior * q1[outcome] for outcome in q0}

    def posterior(self, outcome: Outcome) -> Fraction:
        q1 = self.conditional_observed(1)[outcome]
        return self.prior * q1 / self.observed_marginal()[outcome]

    def is_mnar(self) -> bool:
        return all(by_raw[0] != by_raw[1] for by_raw in self.observe_given_raw)


MODEL_A = LatentConstruction(
    name="A_chain",
    graph=((0, 1), (1, 2)),
    prior=F(1, 2),
    raw_positive=(F(1, 5), F(4, 5)),
    observe_given_raw=((F(3, 4), F(1, 2)), (F(1, 2), F(3, 4))),
)

MODEL_B = LatentConstruction(
    name="B_fork",
    graph=((0, 1), (0, 2)),
    prior=F(1, 2),
    raw_positive=(F(3, 10), F(7, 10)),
    observe_given_raw=((F(5, 7), F(2, 3)), (F(2, 3), F(5, 7))),
)


def _posterior_from_likelihood_ratio(prior: Fraction, ratio: Fraction) -> Fraction:
    return prior * ratio / ((1 - prior) + prior * ratio)


def sharp_likelihood_ratio_bounds(
    observed: Mapping[Outcome, Fraction],
    prior: Fraction,
    likelihood_ratio_bounds: Mapping[Outcome, tuple[Fraction, Fraction]],
) -> dict[Outcome, tuple[Fraction, Fraction]]:
    """Sharp posterior bounds for a finite observed alphabet.

    The uncertainty class fixes P(O=o), P(Z=1), and a lower/upper envelope for
    P(O=o|Z=1)/P(O=o|Z=0). Bounds include the global prior-mass constraint.
    """
    if not F(0) < prior < F(1):
        raise ValueError("prior must lie strictly between zero and one")
    if set(observed) != set(likelihood_ratio_bounds):
        raise ValueError("every observed outcome needs one likelihood-ratio interval")
    if any(value <= 0 for value in observed.values()) or sum(observed.values()) != 1:
        raise ValueError("observed probabilities must be positive and sum to one")

    joint_intervals: dict[Outcome, tuple[Fraction, Fraction]] = {}
    for outcome, probability in observed.items():
        lower_ratio, upper_ratio = likelihood_ratio_bounds[outcome]
        if lower_ratio < 0 or upper_ratio < lower_ratio:
            raise ValueError("invalid likelihood-ratio interval")
        lower_posterior = _posterior_from_likelihood_ratio(prior, lower_ratio)
        upper_posterior = _posterior_from_likelihood_ratio(prior, upper_ratio)
        joint_intervals[outcome] = (
            probability * lower_posterior,
            probability * upper_posterior,
        )

    if sum(value[0] for value in joint_intervals.values()) > prior:
        raise ValueError("lower likelihood-ratio bounds are infeasible")
    if sum(value[1] for value in joint_intervals.values()) < prior:
        raise ValueError("upper likelihood-ratio bounds are infeasible")

    result = {}
    for outcome, probability in observed.items():
        local_lower, local_upper = joint_intervals[outcome]
        other_lower = sum(value[0] for key, value in joint_intervals.items() if key != outcome)
        other_upper = sum(value[1] for key, value in joint_intervals.items() if key != outcome)
        sharp_joint_lower = max(local_lower, prior - other_upper)
        sharp_joint_upper = min(local_upper, prior - other_lower)
        result[outcome] = (sharp_joint_lower / probability, sharp_joint_upper / probability)
    return result


def exact_witness() -> dict[str, object]:
    """Return the exact facts used in the accompanying theorem note."""
    return {
        "model_a_observed": MODEL_A.observed_marginal(),
        "model_b_observed": MODEL_B.observed_marginal(),
        "model_a_posterior_111": MODEL_A.posterior("111"),
        "model_b_posterior_111": MODEL_B.posterior("111"),
        "model_a_posterior_000": MODEL_A.posterior("000"),
        "model_b_posterior_000": MODEL_B.posterior("000"),
        "different_graphs": MODEL_A.graph != MODEL_B.graph,
        "both_mnar": MODEL_A.is_mnar() and MODEL_B.is_mnar(),
    }
