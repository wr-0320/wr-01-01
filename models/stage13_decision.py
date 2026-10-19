"""Small decision-only network for Stage 13 imitation."""

from torch import nn


ACTION_NAMES = ("observe", "track", "commit_A", "commit_B")
INPUT_NAMES = (
    "belief_A",
    "belief_B",
    "entropy",
    "counterfactual_log_likelihood_ratio",
    "cumulative_observation_cost",
    "commitment_status",
)


class Stage13DecisionMLP(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.network = nn.Sequential(
            nn.Linear(len(INPUT_NAMES), self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, len(ACTION_NAMES)),
        )

    def forward(self, state):
        return self.network(state)

