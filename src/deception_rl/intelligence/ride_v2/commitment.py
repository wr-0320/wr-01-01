"""Controllers sharing frozen RCL inference; no simulator state is accepted."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from deception_rl.envs.deceptive_intelligence import BLUE_WAIT,NUM_HYPOTHESES,NUM_SOURCES,NUM_REGIONS,encode_commit_action,encode_verify_action,encode_track_action
from deception_rl.intelligence.adapter import audit_blue_observation


class FrozenRCLCommitment:
    privileged_diagnostic_only=False

    def __init__(self,inference,mode,*,threshold=.85,commit_step=4,cost_weight=1.,rollouts=32,max_cycles=10,seed=0):
        if mode not in {'fixed','confidence','entropy','entropy_verify','risk','risk_passive'}:
            raise ValueError('Unknown control mode')
        self.inference=inference
        self.mode=mode
        self.threshold=float(threshold)
        self.commit_step=int(commit_step)
        self.cost_weight=float(cost_weight)
        self.rollouts=int(rollouts)
        self.max_cycles=int(max_cycles)
        self.seed=int(seed)
        self.name=mode
        self.reset()

    def reset(self):
        self.inference.reset()
        self.info=None
        self.verified=[]

    def _cell_scores(self,belief):
        means=self.inference.likelihood.reliable_mean[...,0]
        average=np.sum(belief[:,None,None]*means,axis=0)
        return np.sum(belief[:,None,None]*(means-average[None])**2,axis=0)

    def _legal_information(self,observation,belief):
        mask=np.asarray(observation['action_mask'],dtype=bool)
        scores=self._cell_scores(belief)
        candidates=[(s,r) for s in range(NUM_SOURCES) for r in range(NUM_REGIONS) if mask[encode_verify_action(s,r)] and (s,r) not in self.verified]
        if candidates:
            cell=max(candidates,key=lambda x:float(scores[x]))
            return encode_verify_action(*cell),cell
        return BLUE_WAIT,None

    def _forecast_risks(self,observation,belief,step):
        """One-step observation-model rollout, not a rollout of the live simulator.

        Sampling assumptions (including intervention effects) are declared in the
        study protocol. Inference likelihood parameters are never changed.
        """
        model=self.inference.likelihood
        mean=model.reliable_mean.reshape(NUM_HYPOTHESES,9,2)
        scale=model.reliable_scale.reshape(NUM_HYPOTHESES,9,2)
        cmean=model.compromised_mean.reshape(9,2)
        cscale=model.compromised_scale.reshape(9,2)
        reliable=self.inference.predictor.reliable.reshape(9)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,step,901]))
        latent=rng.choice(NUM_HYPOTHESES,size=self.rollouts,p=belief)
        uniform=rng.random((self.rollouts,9))
        coverage_draw=rng.random((self.rollouts,9))
        noise=rng.normal(size=(self.rollouts,9,2))
        # Fixed proxy assumptions: independent reports, public sensor coverage,
        # approximate independent-verification and tracking effects.
        base_coverage=np.repeat(np.array([.82,.90,.70]),3)
        actions=[(BLUE_WAIT,.15,None,None)]
        mask=np.asarray(observation['action_mask'],dtype=bool)
        if self.mode=='risk':
            for s in range(3):
                for r in range(3):
                    if mask[encode_verify_action(s,r)] and (s,r) not in self.verified:
                        actions.append((encode_verify_action(s,r),.45,s,r))
            if not np.asarray(observation['tracked_region']).any():
                for r in range(3):
                    if mask[encode_track_action(r)]:actions.append((encode_track_action(r),.30,None,r))
        result={}
        for action,cost,source,region in actions:
            rho=reliable.copy();coverage=base_coverage.copy()
            noise_factor=np.ones((9,2));confidence_offset=np.zeros(9)
            if source is not None:
                cell=source*3+region
                rho[cell]=1.-.30*(1.-rho[cell])
                coverage[cell]=1.
                noise_factor[cell,0]=.35
                confidence_offset[cell]=.10
            elif region is not None:
                cells=np.array([s*3+region for s in range(3)])
                rho[cells]=1.-.65*(1.-rho[cells])
                coverage[cells]=np.minimum(1.,coverage[cells]+.16)
                noise_factor[cells,0]=.65
                confidence_offset[cells]=.04
            normal=uniform<rho[None]
            generating_mean=np.where(normal[...,None],mean[latent],cmean[None])
            generating_scale=np.where(normal[...,None],scale[latent],cscale[None])*noise_factor
            value=generating_mean+generating_scale*noise
            value[...,1]+=confidence_offset
            value=np.clip(value,0.,1.)
            visible=coverage_draw<coverage[None]
            difference=(value[:,None]-mean[None])/scale[None]
            reliable_log=-.5*np.sum(difference**2,axis=-1)-np.sum(np.log(scale),axis=-1)[None]
            corrupted_log=-.5*np.sum(((value-cmean[None])/cscale[None])**2,axis=-1)-np.sum(np.log(cscale),axis=-1)[None]
            mixture=np.logaddexp(np.log(reliable)[None,None]+reliable_log,np.log1p(-reliable)[None,None]+corrupted_log[:,None])
            future=np.log(np.clip(belief,1e-12,1))[None]+np.sum(mixture*visible[:,None],axis=-1)/model.temperature
            future-=future.max(axis=1,keepdims=True)
            posterior=np.exp(future);posterior/=posterior.sum(axis=1,keepdims=True)
            confidence=posterior.max(axis=1)
            # Same correct/incorrect utility and urgency scale as the declared task.
            future_commit=20.*(1.-confidence)+.25*(step+1)*confidence
            result[action]=float(future_commit.mean()+self.cost_weight*cost)
        return result

    def act(self,observation):
        audit_blue_observation(observation)
        self.info=self.inference.infer(observation)
        belief=np.asarray(self.info['hypothesis_belief'])
        predicted=int(belief.argmax())
        step=int(np.asarray(observation['current_step'])[0])
        commit=encode_commit_action(predicted)
        if step>=self.max_cycles-1:return commit
        if self.mode=='fixed':return commit if step>=self.commit_step else BLUE_WAIT
        if self.mode=='confidence':
            return commit if step>=2 and float(belief.max())>=self.threshold else BLUE_WAIT
        if self.mode in {'entropy','entropy_verify'}:
            if step>=2 and self.info['entropy']<=self.threshold:return commit
            if self.mode=='entropy_verify':
                action,cell=self._legal_information(observation,belief)
                if cell is not None:self.verified.append(cell)
                return action
            return BLUE_WAIT
        if step<2:return BLUE_WAIT
        risks=self._forecast_risks(observation,belief,step)
        current=20.*(1.-float(belief.max()))+.25*step*float(belief.max())
        risks[commit]=current
        action=min(risks,key=risks.get)
        self.info['predicted_action_risks']={str(k):v for k,v in risks.items()}
        if action in [encode_verify_action(s,r) for s in range(3) for r in range(3)]:
            index=action-encode_verify_action(0,0)
            self.verified.append(divmod(index,3))
        return action
