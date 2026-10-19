"""Rebuild scientific figures from the frozen summary tables."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'summary.csv').open()))
lookup={(r['method'],r['scope']):r for r in rows}
methods=['multi_source','ride_v1','rcl_fixed','single_mechanism','bayesian_fusion','pomcp_like','active_inference','rule_active']
labels=['Multi-source','RIDe-v1','RCL fixed','Single mechanism','Bayesian','POMCP-like','Active Inference','Rule active']
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for index,metric in enumerate(['correct_commitment','hypothesis_brier']):
    means=np.array([float(lookup[(m,'overall')][metric]) for m in methods])
    lo=np.array([float(lookup[(m,'overall')][metric+'_ci95_low']) for m in methods])
    hi=np.array([float(lookup[(m,'overall')][metric+'_ci95_high']) for m in methods])
    scale=100 if index==0 else 1
    ax=axes[index]
    colors=['#1976a3' if m=='rcl_fixed' else '#a8b4bd' for m in methods]
    ax.barh(np.arange(len(methods)),means*scale,color=colors)
    ax.errorbar(means*scale,np.arange(len(methods)),xerr=np.stack((means-lo,hi-means))*scale,fmt='none',ecolor='#263238',capsize=3)
    ax.set_yticks(np.arange(len(methods)),labels)
    ax.invert_yaxis()
    ax.set_xlabel('Correct commitment (%)' if index==0 else 'Hypothesis Brier (lower is better)')
    ax.grid(axis='x',alpha=.2)
fig.suptitle('Fixed-prior RCL formal evaluation: five fresh training seeds\nError bars: 95% t intervals over seed means',fontsize=11)
fig.tight_layout()
for suffix in ['png','pdf']: fig.savefig(ROOT/f'formal_overview.{suffix}',dpi=180)
plt.close(fig)
families=['honest','concealment','decoy','multi_source','delayed','switch','adaptive']
fig,ax=plt.subplots(figsize=(10,4.4))
x=np.arange(len(families))
for offset,(method,label,color) in enumerate([('multi_source','Multi-source','#a8b4bd'),('single_mechanism','Single mechanism','#e3ad62'),('rcl_fixed','RCL fixed','#1976a3')]):
    values=[100*float(lookup[(method,f)]['correct_commitment']) for f in families]
    ax.bar(x+(offset-1)*.24,values,width=.24,label=label,color=color)
ax.set_xticks(x,['Honest','Concealment','Decoy','Multi-source','Delayed','Switch','Adaptive'])
ax.set_ylabel('Correct commitment (%)')
ax.set_ylim(0,100)
ax.legend(ncol=3,loc='upper left')
ax.grid(axis='y',alpha=.2)
ax.set_title('Deception-family breakdown: 500 scenarios per method and family')
fig.tight_layout()
for suffix in ['png','pdf']: fig.savefig(ROOT/f'deception_family_breakdown.{suffix}',dpi=180)
plt.close(fig)
