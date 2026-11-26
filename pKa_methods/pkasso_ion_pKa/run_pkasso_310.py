# pKasso macro pKa for every recommender ion (first fragment, neutral form). Anion -> pKa(0->-1), cation -> pKa(+1->0).
import warnings, json, numpy as np, pandas as pd; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.MolStandardize import rdMolStandardize as S
from pkasso import scan_pH
U=S.Uncharger()
I=pd.read_csv('ions_mapped_to_v6.csv')
out=[]
for r in I.itertuples():
    s=r.smiles.split('.')[0]; m=Chem.MolFromSmiles(s); n=Chem.MolToSmiles(U.uncharge(m)); val=np.nan; raw={}
    if not r.perm:
        try:
            sc=scan_pH(n,nthreads=1,tautomer_search=False); raw={int(q):float(v) for q,v in sc.pkas_macro.items()}
            val=raw.get(-1 if r.sign<0 else 0,np.nan)
        except Exception as e: raw={'error':str(e)[:60]}
    out.append(dict(row=r.Index,neutral_smiles=n,pkasso_pKa=val,pkasso_all=json.dumps(raw)))
    print(r.Index,round(val,2) if val==val else val,flush=True)
pd.DataFrame(out).to_csv('pkasso_310_ions.csv',index=False)
