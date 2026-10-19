"""Training-free and smoke-scale external Defender baselines."""

from deception_rl.external_baselines.policies import (
    ActiveInferenceDefender,
    BayesianDefenderAdapter,
    MAPPOActorCritic,
    MAPPODefenderPolicy,
    POMCPBeliefPlanner,
)
from deception_rl.external_baselines.benchmark_policies import (
    RIDeDefenderAdapter,
    RuleActiveDefender,
)

__all__ = [
    "ActiveInferenceDefender",
    "BayesianDefenderAdapter",
    "MAPPOActorCritic",
    "MAPPODefenderPolicy",
    "POMCPBeliefPlanner",
    "RIDeDefenderAdapter",
    "RuleActiveDefender",
]
