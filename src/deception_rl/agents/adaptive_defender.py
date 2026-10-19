"""Cost-aware sensing over visible-state reachable sets; frozen belief inference."""
from dataclasses import dataclass
import numpy as np
from deception_rl.envs.gridworld import OBSERVE, TRACK
from deception_rl.agents.decoupled_defender import DecoupledAntiDeception


@dataclass(frozen=True)
class AdaptiveObservationScorer:
    observe_cost: float = .01
    track_cost: float = .03
    false_commitment_penalty: float = .5
    information_gain: float = .08
    observe_radius: int = 2
    track_radius: int = 4

    def __post_init__(self):
        if min(self.observe_cost, self.track_cost, self.false_commitment_penalty, self.information_gain) < 0:
            raise ValueError('Costs, penalties and information gain must be nonnegative')
        if not 0 <= self.observe_radius <= self.track_radius:
            raise ValueError('Invalid sensor radii')

    @staticmethod
    def reachable(position, steps, shape):
        """Public movement envelope; no actual Deceiver action or family access."""
        rows, cols = np.indices(shape)
        mask = np.abs(rows-position[0]) + np.abs(cols-position[1]) <= steps
        return np.column_stack((rows[mask], cols[mask]))

    def evaluate(self, belief, observation, timestep):
        goal = np.asarray(belief['goal_belief'], dtype=float)
        trajectory = belief['trajectory']
        last = np.asarray(trajectory[-1][1:])
        age = max(0, timestep-int(trajectory[-1][0]))
        positions = self.reachable(last, age+1, observation['local_map'].shape)
        sensor_distances = np.abs(positions-np.asarray(observation['sensor_origin'])).sum(axis=1)
        coverage = {'observe': float(np.mean(sensor_distances <= self.observe_radius)),
                    'track': float(np.mean(sensor_distances <= self.track_radius))}
        targets = np.asarray(observation['target_positions'])
        arrival_risk = float(goal[np.abs(targets-last).sum(axis=1) <= age+1].sum())
        # Positive log likelihood-ratio supplies bounded additional demand for information.
        # LR=1 is neutral; these are compatibility scores, not exact generative densities.
        log_lr = max(0., float(np.log(max(belief['likelihood_ratio'], 1e-8))))
        lr_evidence = log_lr/(1+log_lr)
        demand = belief['entropy']*(1+belief['deception_probability']+lr_evidence)
        confidence = float(goal.max())
        utilities = {'commit_A': float(goal[0]-self.false_commitment_penalty*(1-goal[0])),
                     'commit_B': float(goal[1]-self.false_commitment_penalty*(1-goal[1]))}
        for action, cost in [('observe',self.observe_cost), ('track',self.track_cost)]:
            future_correct = min(1., confidence+self.information_gain*demand*coverage[action])
            utilities[action] = float(future_correct*(1-arrival_risk)-cost
                                      -self.false_commitment_penalty*(1-future_correct))
        return utilities, dict(sensor_coverage=coverage, likelihood_information=lr_evidence,
                               information_demand=float(demand), arrival_risk=arrival_risk,
                               visible_age=age, reachable_states=len(positions))


class AdaptiveActiveDefender(DecoupledAntiDeception):
    """Reuse frozen Stage10.1 inference, replacing only observation/action selection."""
    def __init__(self, inference, scorer=None):
        super().__init__(inference)
        self.adaptive_scorer = scorer or AdaptiveObservationScorer()

    def act(self, observation, timestep):
        self.info = self.inference.predict(observation,timestep)
        if self.goal is not None:
            return self.goal
        utilities, diagnostics = self.adaptive_scorer.evaluate(self.info,observation,timestep)
        self.info.update(action_utilities=utilities, adaptive_observation=diagnostics)
        action = max(utilities,key=utilities.get)
        if action.startswith('commit_'):
            self.goal = int(action=='commit_B')
            self.commit_timestep = timestep
            return self.goal
        return OBSERVE if action=='observe' else TRACK
