"""Generic eight-source empirical models for the frozen v3 protocol."""

from __future__ import annotations

import numpy as np

from .generator import ObservedDataset
from .graphs import Graph


def _fit_prior(z: np.ndarray, alpha: float) -> np.ndarray:
    counts = np.bincount(z, minlength=2).astype(float) + alpha
    return np.log(counts / counts.sum())


def _posterior(log_likelihood: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    logits = log_likelihood + log_prior[None, :]
    maximum = logits.max(axis=1, keepdims=True)
    normalized = logits - maximum
    log_norm = maximum[:, 0] + np.log(np.exp(normalized).sum(axis=1))
    return np.exp(logits[:, 1] - log_norm)


class NaiveModel:
    def __init__(self, alpha: float = 1.0): self.alpha = alpha

    def fit(self, data: ObservedDataset):
        self.log_prior = _fit_prior(data.z, self.alpha)
        self.table = np.empty((2, data.y.shape[1]), dtype=float)
        for state in (0, 1):
            for source in range(data.y.shape[1]):
                values = data.y[(data.z == state) & (data.y[:, source] >= 0), source]
                self.table[state, source] = (values.sum() + self.alpha) / (
                    len(values) + 2 * self.alpha
                )
        return self

    def predict_proba(self, y):
        score = np.zeros((len(y), 2), dtype=float)
        for state in (0, 1):
            for source in range(y.shape[1]):
                use = y[:, source] >= 0; value = y[use, source]
                p = self.table[state, source]
                score[use, state] += value * np.log(p) + (1-value) * np.log(1-p)
        return _posterior(score, self.log_prior)


class MissingAwareModel:
    def __init__(self, alpha: float = 1.0): self.alpha = alpha

    def fit(self, data: ObservedDataset):
        self.log_prior = _fit_prior(data.z, self.alpha)
        sources = data.y.shape[1]; encoded = data.y + 1
        self.table = np.empty((2, sources, 3), dtype=float)
        for state in (0, 1):
            for source in range(sources):
                counts = np.bincount(encoded[data.z == state, source], minlength=3).astype(float) + self.alpha
                self.table[state, source] = counts / counts.sum()
        return self

    def predict_proba(self, y):
        encoded=y+1; rows=np.arange(len(y)); score=np.zeros((len(y),2),float)
        for state in (0,1):
            for source in range(y.shape[1]):
                score[:,state] += np.log(self.table[state,source,encoded[:,source]])
        return _posterior(score,self.log_prior)


class DependenceModel:
    def __init__(self, graph: Graph, alpha: float = 1.0):
        self.graph=graph; self.alpha=alpha

    def fit(self,data: ObservedDataset):
        self.log_prior=_fit_prior(data.z,self.alpha)
        self.tables=np.empty((2,len(self.graph),2,2),float)
        for state in (0,1):
            state_y=data.y[data.z==state]
            for index,(a,b) in enumerate(self.graph):
                pair=state_y[:,(a,b)]; complete=pair[(pair>=0).all(axis=1)]
                counts=np.full((2,2),self.alpha,float)
                if len(complete): np.add.at(counts,(complete[:,0],complete[:,1]),1.0)
                self.tables[state,index]=counts/counts.sum()
        return self

    def predict_proba(self,y):
        score=np.zeros((len(y),2),float)
        for state in (0,1):
            for index,(left,right) in enumerate(self.graph):
                a,b=y[:,left],y[:,right]; table=self.tables[state,index]
                probability=np.ones(len(y),float)
                both=(a>=0)&(b>=0); only_a=(a>=0)&(b<0); only_b=(a<0)&(b>=0)
                probability[both]=table[a[both],b[both]]
                probability[only_a]=table.sum(axis=1)[a[only_a]]
                probability[only_b]=table.sum(axis=0)[b[only_b]]
                score[:,state]+=np.log(probability)
        return _posterior(score,self.log_prior)


class JointModel:
    def __init__(self, graph: Graph, alpha: float = 1.0):
        self.graph=graph; self.alpha=alpha

    def fit(self,data: ObservedDataset):
        self.log_prior=_fit_prior(data.z,self.alpha); encoded=data.y+1
        self.tables=np.empty((2,len(self.graph),3,3),float)
        for state in (0,1):
            state_y=encoded[data.z==state]
            for index,(a,b) in enumerate(self.graph):
                pair=state_y[:,(a,b)]; counts=np.full((3,3),self.alpha,float)
                np.add.at(counts,(pair[:,0],pair[:,1]),1.0)
                self.tables[state,index]=counts/counts.sum()
        return self

    def predict_proba(self,y):
        encoded=y+1; score=np.zeros((len(y),2),float)
        for state in (0,1):
            for index,(a,b) in enumerate(self.graph):
                score[:,state]+=np.log(self.tables[state,index,encoded[:,a],encoded[:,b]])
        return _posterior(score,self.log_prior)


class FixedComposition:
    def __init__(self,m1,m2,m3): self.m1=m1; self.m2=m2; self.m3=m3

    @staticmethod
    def _logit(p):
        p=np.clip(p,1e-12,1-1e-12); return np.log(p)-np.log1p(-p)

    def predict_proba(self,y):
        value=self._logit(self.m2.predict_proba(y))+self._logit(self.m3.predict_proba(y))-self._logit(self.m1.predict_proba(y))
        return 1/(1+np.exp(-np.clip(value,-40,40)))


def binary_nll(z, probability):
    p=np.clip(probability,1e-12,1-1e-12)
    return -float(np.mean(z*np.log(p)+(1-z)*np.log(1-p)))

