from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
rows=list(csv.DictReader((p/'summary.csv').open()))
lookup={(r['method'],r['scope']):r for r in rows}
methods=['single_mechanism','standard_rcl','surface_independent','intent_coupled_shared_temperature']
labels=['Single mechanism','Standard RCL','Surface mixture\n(no intent coupling)','Intent-coupled RCL']
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for index,metric in enumerate(['correct_commitment','hypothesis_brier']):
 means=np.array([float(lookup[m,'overall'][metric]) for m in methods])
 lo=np.array([float(lookup[m,'overall'][metric+'_ci95_low']) for m in methods])
 hi=np.array([float(lookup[m,'overall'][metric+'_ci95_high']) for m in methods])
 scale=100 if index==0 else 1
 axes[index].bar(np.arange(4),means*scale,color=['#b9c2c8','#80a8bd','#e3ad62','#1976a3'])
 axes[index].errorbar(np.arange(4),means*scale,yerr=np.stack((means-lo,hi-means))*scale,fmt='none',capsize=3,color='#263238')
 axes[index].set_xticks(np.arange(4),labels,fontsize=8,rotation=12,ha='right')
 axes[index].set_ylim(0,100 if index==0 else .35)
 axes[index].set_ylabel('Correct commitment (%)' if index==0 else 'Hypothesis Brier (lower is better)')
 axes[index].grid(axis='y',alpha=.2)
fig.suptitle('Intent-coupled RCL: three fresh development seeds\n95% t intervals over seed means; seven primary families',fontsize=11)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(p/f'intent_coupled.{ext}',dpi=180)
plt.close(fig)
