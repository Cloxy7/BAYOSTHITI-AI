# Score pKasso (MolGpKa micro-pKa + coupling -> macro pKa) on our 138 measured ions.
# Anion ions are stored as the neutral acid: take macro pKa for charge 0 -> -1 (key -1).
# Cation ions are stored as the neutral base: take macro pKa for charge +1 -> 0 (key 0).
import json, numpy as np, pandas as pd
R=pd.read_csv('/tmp/claude-0/pkrun/pkasso_macro_raw.csv')
def pick(r):
    m={int(k):v for k,v in json.loads(r.macro).items()}
    k=-1 if r.ion_type=='anion' else 0
    return m.get(k,np.nan)
R['pkasso_pKa']=R.apply(pick,axis=1)
R[['ID','smiles','ion_type','pKa_true','pkasso_pKa']].to_csv('../../pKa_QupKake_pKasso/pkasso_predictions_138.csv',index=False)
ok=R.dropna(subset=['pkasso_pKa'])
def sc(t,p): e=p-t; return dict(n=len(t),MAE=round(np.abs(e).mean(),3),RMSE=round(np.sqrt((e**2).mean()),3),R2=round(1-(e**2).sum()/((t-t.mean())**2).sum(),3))
print('predicted for',len(ok),'of',len(R))
for lab,s in [('all',ok),('anions',ok[ok.ion_type=='anion']),('cations',ok[ok.ion_type=='cation'])]: print(lab,sc(s.pKa_true.values,s.pkasso_pKa.values))
