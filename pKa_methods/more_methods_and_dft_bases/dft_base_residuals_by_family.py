# Why is the DFT line weak for bases? Leave-one-out residuals by chemical family (same data and filter as dft_base_chemotype_fix.py)
import re, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
src=open('dft_base_chemotype_fix.py').read().split("y=D.pKa.values")[0]   # reuse loading + chemotype labelling
exec(src)
b=D[D.kind=='base'].reset_index(drop=True); y=b.pKa.values; x=b.dE.values; pred=np.zeros(len(b))
for i in range(len(b)):
    t=np.arange(len(b))!=i; k,c=np.polyfit(x[t],y[t],1); pred[i]=k*x[i]+c
b['pred']=pred; b['resid']=pred-y
print('pKa range of bases: %.1f to %.1f, sd %.2f; dE vs pKa Pearson r = %.2f'%(y.min(),y.max(),y.std(),np.corrcoef(x,y)[0,1]))
g=b.groupby('chemotype').agg(n=('pKa','size'),pKa_mean=('pKa','mean'),pKa_min=('pKa','min'),pKa_max=('pKa','max'),MAE=('resid',lambda r:np.abs(r).mean()),bias=('resid','mean')).round(2).sort_values('MAE',ascending=False)
print(g.to_string()); g.to_csv('dft_base_errors_by_family.csv')
# how much of the error is a per-family bias, and what is left
b['resid_after']=b.resid-b.groupby('chemotype').resid.transform('mean')
print('MAE all %.2f | after removing each family bias (optimistic, in-sample) %.2f'%(np.abs(b.resid).mean(),np.abs(b.resid_after).mean()))
worst=b.reindex(b.resid.abs().sort_values(ascending=False).index).head(8)[['name','chemotype','pKa','pred','resid']].round(2); print(worst.to_string(index=False)); worst.to_csv('dft_base_worst8.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(12,4.8)); cols=dict(zip(sorted(b.chemotype.unique()),plt.cm.tab10.colors))
for t,s in b.groupby('chemotype'):
    ax[0].scatter(s.pKa,s.pred,label=f'{t} (n={len(s)})',color=cols[t],s=45)
lim=[min(y.min(),pred.min())-0.5,max(y.max(),pred.max())+0.5]; ax[0].plot(lim,lim,'k--',lw=1); ax[0].set_xlabel('measured pKa'); ax[0].set_ylabel('DFT-line pKa (leave-one-out)'); ax[0].set_title('Bases: DFT line vs measured'); ax[0].legend(fontsize=7)
for t,s in b.groupby('chemotype'): ax[1].scatter(s.dE,s.pKa,color=cols[t],s=45)
ax[1].set_xlabel('DFT energy dE'); ax[1].set_ylabel('measured pKa'); ax[1].set_title('One line has to fit every family')
plt.tight_layout(); plt.savefig('dft_base_by_family.png',dpi=150)
