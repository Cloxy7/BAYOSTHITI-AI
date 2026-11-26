# Can a chemotype-aware calibration of the DFT energy fix the weak base results? (leave-one-out, our measured ions that have a DFT energy)
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from sklearn.linear_model import Ridge
O='../../'; U='/root/.claude/uploads/cd16bc9d-cde6-538b-bfce-1e53117b19ea/'
rf=pd.read_csv(O+'IL_ion_pKa_ML_meets_pKa.csv')[['ion_id','name','ion_type','smiles','pKa','pKa_status','pKa_confidence','ml_pKa']]
dft=pd.read_csv(U+'25bcc783-IL_pKa_DFT_SMD_1.csv')[['master_id','kind','dE']].rename(columns={'master_id':'ion_id'})
D=rf.merge(dft,on='ion_id'); conf=D.pKa_confidence.fillna('').str.lower(); D=D[(D.pKa_status=='measured')|((D.pKa_status=='web lookup')&conf.str.startswith('measured |'))].dropna(subset=['dE']).reset_index(drop=True)
P={'aromatic amine (aniline-type)':'[NX4+;H3,H2,H1]c','amidine/guanidine':'[NX3,NX3+]C(=[NX2,NX3+])[NX3]','N-heteroaromatic (imidazolium/pyridinium)':'[n+,nH+]','carboxylate+amine (amino acid)':'[O-]C(=O)C[NX3,NX4+]','primary/secondary/tertiary aliphatic ammonium':'[NX4+;!$(N-c)]'}
def chemo(smi,kind):
    m=Chem.MolFromSmiles(smi)
    if kind=='acid': return 'acid'
    for k,s in P.items():
        if m.HasSubstructMatch(Chem.MolFromSmarts(s)): return k
    return 'other base'
D['chemotype']=[chemo(s,k) for s,k in zip(D.smiles,D.kind)]
print(D[D.kind=='base'].chemotype.value_counts().to_dict())
y=D.pKa.values; dE=D.dE.values; kind=D.kind.values; ch=D.chemotype.values; n=len(D)
types=sorted(set(ch)); 
def sc(t,p): e=p-t; return dict(n=len(t),MAE=round(np.abs(e).mean(),3),RMSE=round(np.sqrt((e**2).mean()),3),R2=round(1-(e**2).sum()/((t-t.mean())**2).sum(),3))
def X(idx,use_chemo): 
    cols=[dE[idx]]+([ (ch[idx]==t).astype(float) for t in types] if use_chemo else [])
    return np.c_[tuple(cols)]
res={}
for nm,use in [('one line per kind (current)',False),('line + chemotype offset',True)]:
    p=np.zeros(n)
    for i in range(n):
        tr=np.array([j for j in range(n) if j!=i and kind[j]==kind[i]]); m=Ridge(alpha=0.3).fit(X(tr,use),y[tr]); p[i]=m.predict(X(np.array([i]),use))[0]
    res[nm]=p
# add RF as extra input
p=np.zeros(n)
for i in range(n):
    tr=np.array([j for j in range(n) if j!=i and kind[j]==kind[i] and not np.isnan(D.ml_pKa[j])]); 
    A=lambda idx:np.c_[X(idx,True),D.ml_pKa.values[idx]]
    if np.isnan(D.ml_pKa[i]): p[i]=np.nan; continue
    m=Ridge(alpha=0.3).fit(A(tr),y[tr]); p[i]=m.predict(A(np.array([i])))[0]
res['line + chemotype + RF prediction']=p
rows=[]
for nm,p in res.items():
    for k in ('acid','base'):
        mk=(kind==k)&~np.isnan(p); rows.append(dict(method=nm,group=k,**sc(y[mk],p[mk])))
R=pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv('dft_base_chemotype_results.csv',index=False)
