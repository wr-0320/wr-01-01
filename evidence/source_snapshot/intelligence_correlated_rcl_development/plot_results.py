from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
rows=list(csv.DictReader((p/'summary.csv').open()))
lookup={(r['method'],r['scope']):r for r in rows}
methods=['single_mechanism','independent_rcl','correlated_shared_temperature','correlated_calibrated']
labels=['Single mechanism','Independent RCL','Correlated RCL\n(shared temperature)','Correlated RCL\n(recalibrated)']
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for index,metric in enumerate(['correct_commitment','hypothesis_brier']):
 values=np.array([float(lookup[m,'overall'][metric]) for m in methods])
 lower=np.array([float(lookup[m,'overall'][metric+'_ci95_low']) for m in methods])
 upper=np.array([float(lookup[m,'overall'][metric+'_ci95_high']) for m in methods])
 scale=100 if index==0 else 1
 axes[index].bar(np.arange(4),values*scale,color=['#a8b4bd','#80a8bd','#1976a3','#e3ad62'])
 axes[index].errorbar(np.arange(4),values*scale,yerr=np.stack((values-lower,upper-values))*scale,fmt='none',capsize=3,color='#263238')
 axes[index].set_xticks(np.arange(4),labels,rotation=15,ha='right',fontsize=8)
 axes[index].set_ylabel('Correct commitment (%)' if index==0 else 'Hypothesis Brier (lower is better)')
 axes[index].set_ylim(0,100 if index==0 else .35)
 axes[index].grid(axis='y',alpha=.2)
fig.suptitle('Shared contamination latent: three fresh development seeds\nError bars: 95% t intervals over seed means',fontsize=11)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(p/f'correlated_rcl.{ext}',dpi=180)
plt.close(fig)
