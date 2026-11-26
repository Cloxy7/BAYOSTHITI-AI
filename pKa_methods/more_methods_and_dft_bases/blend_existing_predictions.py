# Blend RF, GNN, MolGpKa and a DFT line on our measured ions. DFT is rebuilt from the raw deprotonation energy dE with a
# calibration line per kind (acid/base) fitted WITHOUT the ion being predicted (leave-one-out), and nested for the blend.
import numpy as np, pandas as pd
O='../../'; U='/root/.claude/uploads/cd16bc9d-cde6-538b-bfce-1e53117b19ea/'
rf=pd.read_csv(O+'IL_ion_pKa_ML_meets_pKa.csv')[['ion_id','ion_type','pKa','pKa_status','pKa_confidence','ml_pKa']]
gnn=pd.read_csv(O+'IL_ion_pKa_GraphPKa_GNN.csv')[['ion_id','gnn_pKa']]
dft=pd.read_csv(U+'25bcc783-IL_pKa_DFT_SMD_1.csv')[['master_id','kind','dE']].rename(columns={'master_id':'ion_id'})
mg=pd.read_csv(O+'pKa_more_methods/molgpka_predictions_on_our_ions.csv')[['ion_id','molgpka_pKa']]
conf=rf.pKa_confidence.fillna('').str.lower()
D=rf[(rf.pKa_status=='measured')|((rf.pKa_status=='web lookup')&conf.str.startswith('measured |'))].merge(gnn,on='ion_id',how='left').merge(dft,on='ion_id',how='left').merge(mg,on='ion_id',how='left').rename(columns={'pKa':'true'})
D=D.dropna(subset=['dE','kind']).reset_index(drop=True); n=len(D); print('measured ions with a DFT energy:',n, D.kind.value_counts().to_dict())
y=D.true.values; dE=D.dE.values; kind=D.kind.values
def line(excl):
    co={}
    for k in ('acid','base'):
        m=np.array([kind[j]==k and j not in excl for j in range(n)]); co[k]=np.polyfit(dE[m],y[m],1)
    return co
dft_loo=np.array([np.polyval(line({i})[kind[i]],dE[i]) for i in range(n)])
def sc(t,p): e=p-t; return dict(n=len(t),MAE=round(np.abs(e).mean(),3),RMSE=round(np.sqrt((e**2).mean()),3),R2=round(1-(e**2).sum()/((t-t.mean())**2).sum(),3))
print('DFT line, leave-one-out: acids',sc(y[kind=='acid'],dft_loo[kind=='acid']),'bases',sc(y[kind=='base'],dft_loo[kind=='base']),'all',sc(y,dft_loo))
from sklearn.linear_model import Ridge
C=D.dropna(subset=['ml_pKa','gnn_pKa','molgpka_pKa']).index.values; print('ions with RF, GNN, MolGpKa and DFT:',len(C))
# nested LOO stack: for held-out i, the DFT value of every training ion j is computed with a line fitted without i and j
def nested_stack(cols):
    out={}
    for i in C:
        tr=[j for j in C if j!=i]; Xtr=[];ytr=[]
        for j in tr:
            c=line({i,j}); d=np.polyval(c[kind[j]],dE[j]); row={'rf':D.ml_pKa[j],'gnn':D.gnn_pKa[j],'dft':d,'mg':D.molgpka_pKa[j]}; Xtr.append([row[c_] for c_ in cols]); ytr.append(y[j])
        c=line({i}); d=np.polyval(c[kind[i]],dE[i]); row={'rf':D.ml_pKa[i],'gnn':D.gnn_pKa[i],'dft':d,'mg':D.molgpka_pKa[i]}
        m=Ridge(alpha=1.0).fit(np.array(Xtr),ytr); out[i]=m.predict(np.array([[row[c_] for c_ in cols]]))[0]
    return np.array([out[i] for i in C])
rows=[]; Y=y[C]; K=kind[C]
cand={'RF':D.ml_pKa.values[C],'GNN':D.gnn_pKa.values[C],'MolGpKa':D.molgpka_pKa.values[C],'DFT line (LOO)':dft_loo[C],'average RF + DFT':(D.ml_pKa.values[C]+dft_loo[C])/2}
for nm,cols in [('stack RF + DFT (nested LOO)',['rf','dft']),('stack RF + GNN + MolGpKa + DFT (nested LOO)',['rf','gnn','mg','dft']),('stack RF + GNN + MolGpKa, no DFT (nested LOO)',['rf','gnn','mg'])]:
    if 'dft' not in cols:
        F=lambda j:[{'rf':D.ml_pKa[j],'gnn':D.gnn_pKa[j],'mg':D.molgpka_pKa[j]}[c_] for c_ in cols]
        out=[]
        for i in C:
            tr=[j for j in C if j!=i]; m=Ridge(alpha=1.0).fit(np.array([F(j) for j in tr]),y[tr]); out.append(m.predict(np.array([F(i)]))[0])
        cand[nm]=np.array(out)
    else: cand[nm]=nested_stack(cols)
for lab,mask in [('all',np.ones(len(C),bool)),('anions / acids',K=='acid'),('cations / bases',K=='base')]:
    for nm,p in cand.items(): rows.append(dict(group=lab,method=nm,**sc(Y[mask],p[mask])))
R=pd.DataFrame(rows); R.to_csv('blend_existing_results.csv',index=False); print(R.sort_values(['group','MAE']).to_string(index=False))
