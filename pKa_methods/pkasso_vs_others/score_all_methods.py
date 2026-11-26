# Compare RF, direct MolGpKa, pKasso, QupKake and the DFT line on the same measured ions.
# Anion ions are stored as the neutral acid (pKa of the acid); cation ions as the neutral base (pKa of its conjugate acid).
import json, glob, numpy as np, pandas as pd
from rdkit import Chem
O='../../'; U='/root/.claude/uploads/cd16bc9d-cde6-538b-bfce-1e53117b19ea/'
base=pd.read_csv('/tmp/claude-0/pkaml/our_measured_prepared.csv').rename(columns={'ID':'ion_id'})
pk=pd.read_csv(O+'pKa_QupKake_pKasso/pkasso_predictions_138.csv').rename(columns={'ID':'ion_id'})[['ion_id','pkasso_pKa']]
mg=pd.read_csv('/tmp/claude-0/pkrun/molgpka_prepared.csv').rename(columns={'ID':'ion_id'})[['ion_id','molgpka_prepared']]
rf=pd.read_csv(O+'IL_ion_pKa_ML_meets_pKa.csv')[['ion_id','ml_pKa']]
dft=pd.read_csv(U+'25bcc783-IL_pKa_DFT_SMD_1.csv')[['master_id','kind','dE']].rename(columns={'master_id':'ion_id'})
# QupKake: SDF records carry the molecule name, atom index and predicted pKa per site
qs=[]
from rdkit.Chem import SDMolSupplier
for m in SDMolSupplier('/tmp/claude-0/qrun/data/output/q138.sdf',removeHs=False):
    if m is None: continue
    qs.append(dict(ion_id=m.GetProp('_Name'),pka_type=m.GetProp('pka_type'),pka=float(m.GetProp('pka'))))
Q=pd.DataFrame(qs); Q.to_csv(O+'pKa_QupKake_pKasso/qupkake_all_site_predictions.csv',index=False)
def qpick(g,t): 
    x=g[g.pka_type==('acidic' if t=='anion' else 'basic')].pka
    return (x.min() if t=='anion' else x.max()) if len(x) else np.nan
base['qupkake_pKa']=[qpick(Q[Q.ion_id==r.ion_id],r.ion_type) for r in base.itertuples()]
D=base.merge(pk,on='ion_id',how='left').merge(mg,on='ion_id',how='left').merge(rf,on='ion_id',how='left').merge(dft,on='ion_id',how='left')
# DFT line: refit per kind WITHOUT the ion itself (leave-one-out) using raw dE
E_=D.dropna(subset=['dE','kind']).reset_index(drop=True); y=E_.pKa_true.values; dE=E_.dE.values; kd=E_.kind.values; n=len(E_)
loo=[]
for i in range(n):
    m=np.array([kd[j]==kd[i] and j!=i for j in range(n)]); loo.append(np.polyval(np.polyfit(dE[m],y[m],1),dE[i]))
E_['dft_line_LOO']=loo; D=D.merge(E_[['ion_id','dft_line_LOO']],on='ion_id',how='left')
D.to_csv(O+'pKa_QupKake_pKasso/all_methods_predictions_138.csv',index=False)
def sc(t,p): e=p-t; return dict(n=len(t),MAE=round(float(np.abs(e).mean()),2),RMSE=round(float(np.sqrt((e**2).mean())),2),R2=round(float(1-(e**2).sum()/((t-t.mean())**2).sum()),2),within1=round(float((np.abs(e)<=1).mean()),2))
cols={'RF (ML meets pKa)':'ml_pKa','MolGpKa direct':'molgpka_prepared','pKasso (MolGpKa+coupling)':'pkasso_pKa','QupKake':'qupkake_pKa','DFT line (LOO)':'dft_line_LOO'}
rows=[]
sets={'A: all ions each method can predict':None,'B: 69 ions that have a DFT energy (all 5 methods)':'dft'}
for sname,flag in sets.items():
    S=D.copy()
    if flag=='dft': S=S.dropna(subset=list(cols.values()))
    for grp,mask in [('all',S.ion_type.notna()),('anions',S.ion_type=='anion'),('cations',S.ion_type=='cation')]:
        for nm,c in cols.items():
            T=S[mask].dropna(subset=[c])
            if len(T)>=5: rows.append(dict(set=sname,group=grp,method=nm,**sc(T.pKa_true.values,T[c].values)))
R=pd.DataFrame(rows); R.to_csv(O+'pKa_QupKake_pKasso/comparison_results.csv',index=False)
print(R.to_string(index=False))
