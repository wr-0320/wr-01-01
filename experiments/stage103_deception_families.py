"""Fixed robustness-only strategies; never added to frozen inference prototypes."""
import numpy as np
from deception_rl.agents import ScriptedDeception
from experiments.deception_families import _action_toward

FAMILIES = ('scripted_8','scripted_10','delay_detour','detour_switch','stochastic_deception')
MOVES = np.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]])


class MixedDeception:
    def __init__(self, family):
        if family not in ('delay_detour','detour_switch'):
            raise ValueError('Unknown mixed family')
        self.family=family
        self.reset()

    def reset(self,seed=None):
        self.steps=0

    def act(self,observation):
        position=np.asarray(observation['position']);goal=int(observation['goal'])
        targets=np.asarray(observation['target_positions']);t=self.steps;self.steps+=1
        if self.family=='delay_detour':
            # Honest2 -> false-goal2 -> lateral false-side2 -> honest to completion.
            if t<2:target=targets[goal]
            elif t<4:target=targets[1-goal]
            elif t<6:target=np.array([position[0],targets[1-goal,1]])
            else:target=targets[goal]
        else:
            # Lateral false-side2 -> false/true alternating4 -> honest to completion.
            if t<2:target=np.array([position[0],targets[1-goal,1]])
            elif t<6:target=targets[1-goal if (t-2)%2==0 else goal]
            else:target=targets[goal]
        return _action_toward(position,target)


class StochasticDeception:
    def reset(self,seed=None):
        self.steps=0;self.rng=np.random.default_rng(seed)

    def __init__(self):self.reset()

    def act(self,observation):
        t=self.steps;self.steps+=1
        position=np.asarray(observation['position']);goal=int(observation['goal'])
        targets=np.asarray(observation['target_positions'])
        if t>=8:return _action_toward(position,targets[goal])
        draw=self.rng.random()
        if draw<.5:return _action_toward(position,targets[1-goal])
        if draw<.8:return _action_toward(position,targets[goal])
        return int(self.rng.integers(0,5))


def policy(family):
    if family in ('scripted_8','scripted_10'):
        return ScriptedDeception(int(family.rsplit('_',1)[1]))
    if family in ('delay_detour','detour_switch'):return MixedDeception(family)
    if family=='stochastic_deception':return StochasticDeception()
    raise ValueError(f'Unknown robustness family: {family}')


def scenario_schedule(seed=0):
    scenarios=[]
    for family in FAMILIES:
        count=32 if family=='stochastic_deception' else 34
        for i in range(count):
            scenarios.append(dict(family=family,goal='A' if i%2==0 else 'B',
                                  seed=seed+1500000+len(scenarios),episode_id=len(scenarios),group='paired'))
    audits=[dict(family='stochastic_deception',goal='A' if i%2==0 else 'B',
                 seed=seed+1600000+i,episode_id=168+i,group='supplementary') for i in range(8)]
    assert len(scenarios)==168 and len(scenarios)*3+len(audits)==512
    return scenarios,audits
