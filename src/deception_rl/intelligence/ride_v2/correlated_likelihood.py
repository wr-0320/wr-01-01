"""Shared regional contamination latents with matched independent marginals."""
from __future__ import annotations
import numpy as np
from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES,NUM_SOURCES,NUM_REGIONS
from deception_rl.intelligence.adapter import audit_blue_observation


def logsumexp(value,axis):
    maximum=np.max(value,axis=axis,keepdims=True)
    return np.squeeze(maximum+np.log(np.exp(value-maximum).sum(axis=axis,keepdims=True)),axis=axis)


class CorrelationAwareLikelihood:
    """One shared binary event per region/report cycle, independent across cycles.

    Emissions and unconditional cell contamination rates are inherited unchanged
    from the matched independent RCL. Only the dependency structure is changed.
    Fit uses offline mechanism labels; logits accepts only blue-visible reports.
    """
    def __init__(self,base,event_prior,conditional_compromise):
        self.base=base
        self.temperature=base.temperature
        self.event_prior=np.asarray(event_prior,dtype=float).copy()
        self.conditional_compromise=np.asarray(conditional_compromise,dtype=float).copy()
        if self.event_prior.shape!=(NUM_REGIONS,):raise ValueError('Invalid regional prior shape')
        if self.conditional_compromise.shape!=(2,NUM_SOURCES,NUM_REGIONS):raise ValueError('Invalid conditional shape')
        if not np.all((self.event_prior>0)&(self.event_prior<1)):raise ValueError('Noninterior event prior')
        if not np.all((self.conditional_compromise>0)&(self.conditional_compromise<1)):raise ValueError('Noninterior cell rates')
        self.event_prior.setflags(write=False)
        self.conditional_compromise.setflags(write=False)

    @classmethod
    def fit_dependency(cls,samples,base,fixed,*,strength=1.):
        if not 0<=strength<=1:raise ValueError('Strength must be in [0,1]')
        if not samples:raise ValueError('Empty fitting data')
        states=np.stack([np.asarray(r['compromised_cells'],dtype=int) for r in samples])
        masks=np.stack([np.asarray(r['reported_cells'],dtype=bool) for r in samples])
        event=(states.sum(axis=1)>=2).astype(int)
        prior=(event.sum(axis=0)+1.)/(len(event)+2.)
        estimates=np.zeros((2,NUM_SOURCES,NUM_REGIONS))
        for z in range(2):
            for region in range(NUM_REGIONS):
                selected=masks[:,:,region]*(event[:,region]==z)[:,None]
                estimates[z,:,region]=((selected*states[:,:,region]).sum(axis=0)+1.)/(selected.sum(axis=0)+2.)
        target=1.-np.asarray(fixed.reliable,dtype=float)
        maximum=np.minimum(target/prior[None],(1.-target)/(1.-prior)[None])
        slope=np.minimum(np.maximum(estimates[1]-estimates[0],0.),maximum*(1.-1e-8))*strength
        conditional=np.stack((target-prior[None]*slope,target+(1.-prior)[None]*slope))
        value=cls(base,prior,conditional)
        if not np.allclose(value.marginal_compromise(),target,atol=1e-12,rtol=0):raise AssertionError('Marginal contamination changed')
        return value

    def marginal_compromise(self):
        return (1.-self.event_prior)[None]*self.conditional_compromise[0]+self.event_prior[None]*self.conditional_compromise[1]

    def with_temperature(self,temperature):
        return CorrelationAwareLikelihood(self.base.with_temperature(temperature),self.event_prior,self.conditional_compromise)

    def _regional_terms(self,reports,mask):
        reliable=self.base._log_normal(reports[None,...,:2],self.base.reliable_mean,self.base.reliable_scale)
        compromised=self.base._log_normal(reports[...,:2],self.base.compromised_mean,self.base.compromised_scale)
        log_components=[]
        for z in range(2):
            rate=self.conditional_compromise[z]
            cell=np.logaddexp(np.log1p(-rate)[None]+reliable,np.log(rate)[None]+compromised[None])
            regional=(cell*mask[None]).sum(axis=1)
            log_prior=np.log(self.event_prior if z else 1.-self.event_prior)
            log_components.append(regional+log_prior[None])
        return np.stack(log_components)  # z × hypothesis × region

    def logits(self,observation,predictor=None,reasoner=None):
        audit_blue_observation(observation)
        reports=np.asarray(observation['reports'],dtype=float)
        mask=np.asarray(observation['report_mask'],dtype=bool)
        times=np.flatnonzero(np.asarray(observation['time_mask'],dtype=bool))
        value=np.full(NUM_HYPOTHESES,-np.log(NUM_HYPOTHESES))
        for timestep in times:
            terms=self._regional_terms(reports[timestep],mask[timestep])
            value+=logsumexp(terms,axis=0).sum(axis=1)
        return value/self.temperature

    def latest_coordination_scores(self,observation,hypothesis_belief):
        audit_blue_observation(observation)
        times=np.flatnonzero(np.asarray(observation['time_mask'],dtype=bool))
        if not len(times):return self.event_prior.copy()
        t=int(times[-1])
        terms=self._regional_terms(np.asarray(observation['reports'])[t],np.asarray(observation['report_mask'],dtype=bool)[t])
        responsibility=np.exp(terms[1]-logsumexp(terms,axis=0))
        return np.sum(np.asarray(hypothesis_belief)[:,None]*responsibility,axis=0)

    def contamination_log_probability(self,row):
        """Offline joint-label score, used only by the evaluation harness."""
        label=np.asarray(row['compromised_cells'],dtype=int)
        mask=np.asarray(row['reported_cells'],dtype=bool)
        terms=[]
        for z in range(2):
            rates=self.conditional_compromise[z]
            likelihood=label*np.log(rates)+(1.-label)*np.log1p(-rates)
            terms.append((likelihood*mask).sum(axis=0)+np.log(self.event_prior if z else 1.-self.event_prior))
        return float(logsumexp(np.stack(terms),axis=0).sum())

    def to_dict(self):
        return {'event_prior':self.event_prior.tolist(),'conditional_compromise':self.conditional_compromise.tolist(),'marginal_compromise':self.marginal_compromise().tolist(),'temperature':self.temperature,'latent_scope':'region × report cycle','cross_region_and_temporal_dependencies_modeled':False}
