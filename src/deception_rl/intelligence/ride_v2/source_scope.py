"""Consistent single-source views for inference and offline fitting."""
from copy import deepcopy
import numpy as np
import torch
from deception_rl.envs.deceptive_intelligence import NUM_SOURCES,NUM_REGIONS
from deception_rl.intelligence.adapter import audit_blue_observation,CELL_FEATURES


def restrict_observation(observation, source):
    audit_blue_observation(observation)
    if not 0 <= source < NUM_SOURCES:
        raise ValueError('Invalid single-source index')
    result=deepcopy(observation)
    others=[s for s in range(NUM_SOURCES) if s!=source]
    result['reports'][:,others]=0
    result['report_mask'][:,others]=0
    result['source_reliability'][others]=0
    audit_blue_observation(result)
    return result


def restrict_sequence(sequence,source):
    if not 0 <= source < NUM_SOURCES:
        raise ValueError('Invalid single-source index')
    result=torch.as_tensor(sequence).clone()
    for other in range(NUM_SOURCES):
        if other==source: continue
        start=other*NUM_REGIONS*CELL_FEATURES
        result[:,start:start+NUM_REGIONS*CELL_FEATURES]=0
        result[:,NUM_SOURCES*NUM_REGIONS*CELL_FEATURES+other]=0
    return result


def restrict_samples(samples,source):
    if not 0 <= source < NUM_SOURCES:
        raise ValueError('Invalid single-source index')
    result=deepcopy(samples)
    others=[s for s in range(NUM_SOURCES) if s!=source]
    for row in result:
        for name in ('reported_cells','reported_values','compromised_cells','cell_features','features','compromised'):
            if name in row:
                row[name]=np.asarray(row[name]).copy()
                row[name][others]=0
    return result


class SourceScopedController:
    privileged_diagnostic_only=False

    def __init__(self,controller,source,name):
        self.controller=controller
        self.source=source
        self.name=name
        self.info=None

    def reset(self):
        self.controller.reset()
        self.info=None

    def infer(self,observation):
        view=restrict_observation(observation,self.source)
        return self.controller.infer(view)

    def act(self,observation):
        action=self.controller.act(restrict_observation(observation,self.source))
        self.info=self.controller.info
        return action
