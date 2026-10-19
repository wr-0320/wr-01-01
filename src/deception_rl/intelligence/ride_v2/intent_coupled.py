"""Intent-conditioned mixture over latent presentation regions/events."""
from __future__ import annotations
import numpy as np
from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES,NUM_REGIONS,NUM_SOURCES
from deception_rl.intelligence.adapter import audit_blue_observation
from deception_rl.intelligence.ride_v2.correlated_likelihood import logsumexp

SURFACE_STATES=('decoy_north','decoy_central','decoy_south','concealment')


class IntentCoupledLikelihood:
    """Shares reliable emissions and cell rates with standard RCL.

    F is a latent induced region, plus an explicit no-induced-region concealment
    component. A single F is marginalized jointly across visible cells per cycle.
    Component fitting uses observed reports during known presentation events;
    the matched unconditional-F model uses exactly the same component parameters.
    Different report cycles are still treated as conditionally independent.
    """
    def __init__(self,base,reliable,surface_mean,surface_scale,transition,*,coupled=True):
        self.base=base
        self.temperature=base.temperature
        self.reliable=np.asarray(reliable,dtype=float).copy()
        self.surface_mean=np.asarray(surface_mean,dtype=float).copy()
        self.surface_scale=np.asarray(surface_scale,dtype=float).copy()
        self.transition=np.asarray(transition,dtype=float).copy()
        if self.surface_mean.shape!=(4,NUM_SOURCES,NUM_REGIONS,2):raise ValueError('Invalid surface emission shape')
        if self.surface_scale.shape!=self.surface_mean.shape:raise ValueError('Invalid surface scale shape')
        if self.transition.shape!=(NUM_HYPOTHESES,4):raise ValueError('Invalid transition shape')
        if not np.allclose(self.transition.sum(axis=1),1.):raise ValueError('Transition not normalized')
        if not np.all(self.transition>0):raise ValueError('Zero transition probabilities')
        self.coupled=bool(coupled)

    @classmethod
    def fit(cls,rows,base,fixed,*,coupled=True):
        counts=np.ones((NUM_HYPOTHESES,4),dtype=float)
        buckets={(f,s,r):[] for f in range(4) for s in range(NUM_SOURCES) for r in range(NUM_REGIONS)}
        for row in rows:
            surface=row['surface_label']
            if surface is None:continue
            counts[int(row['hypothesis_label']),int(surface)]+=1
            values=np.asarray(row['reported_values'],dtype=float)
            mask=np.asarray(row['reported_cells'],dtype=bool)
            for source,region in np.argwhere(mask):
                buckets[int(surface),int(source),int(region)].append(values[source,region,:2])
        transition=counts/counts.sum(axis=1,keepdims=True)
        means=np.broadcast_to(base.compromised_mean,(4,NUM_SOURCES,NUM_REGIONS,2)).copy()
        scales=np.broadcast_to(base.compromised_scale,means.shape).copy()
        for (surface,source,region),values in buckets.items():
            if len(values)>=3:
                array=np.asarray(values)
                means[surface,source,region]=array.mean(axis=0)
                scales[surface,source,region]=np.maximum(array.std(axis=0),.08)
        return cls(base,fixed.reliable,means,scales,transition,coupled=coupled)

    def with_temperature(self,temperature):
        return IntentCoupledLikelihood(self.base.with_temperature(temperature),self.reliable,self.surface_mean,self.surface_scale,self.transition,coupled=self.coupled)

    def _weights(self):
        if self.coupled:return self.transition
        prior=self.transition.mean(axis=0)
        return np.broadcast_to(prior,self.transition.shape)

    def _cycle_terms(self,reports,mask):
        reliable_log=self.base._log_normal(reports[None,...,:2],self.base.reliable_mean,self.base.reliable_scale)
        surface_log=self.base._log_normal(reports[None,...,:2],self.surface_mean,self.surface_scale)
        cell=np.logaddexp(np.log(self.reliable)[None,None]+reliable_log[None],np.log1p(-self.reliable)[None,None]+surface_log[:,None])
        per_surface=(cell*mask[None,None]).sum(axis=(2,3))  # F x H
        return per_surface+np.log(self._weights().T)

    def logits(self,observation,predictor=None,reasoner=None):
        audit_blue_observation(observation)
        reports=np.asarray(observation['reports'],dtype=float)
        mask=np.asarray(observation['report_mask'],dtype=bool)
        result=np.full(NUM_HYPOTHESES,-np.log(NUM_HYPOTHESES))
        for timestep in np.flatnonzero(np.asarray(observation['time_mask'],dtype=bool)):
            result+=logsumexp(self._cycle_terms(reports[timestep],mask[timestep]),axis=0)
        return result/self.temperature

    def to_dict(self):
        return {'surface_states':list(SURFACE_STATES),'transition':self.transition.tolist(),'surface_mean':self.surface_mean.tolist(),'surface_scale':self.surface_scale.tolist(),'reliable':self.reliable.tolist(),'temperature':self.temperature,'coupled':self.coupled,'latent_scope':'one regional presentation event per report cycle; independent across cycles'}
