# pKasso (MolGpKa micro-pKa + coupling -> macro pKa) on our measured ions
import warnings, sys, json, time; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from pkasso import scan_pH
d=pd.read_csv('/tmp/claude-0/pkaml/our_measured_prepared.csv')
N=int(sys.argv[1]) if len(sys.argv)>1 else len(d)
rows=[]; t0=time.time()
for r in d.head(N).itertuples():
    try:
        s=scan_pH(r.smiles_prepared,nthreads=1,tautomer_search=False)
        mp={int(q):float(v) for q,v in s.pkas_macro.items()}
    except Exception as e:
        mp={}; print('fail',r.ID,str(e)[:80],flush=True)
    rows.append(dict(ID=r.ID,smiles=r.smiles_prepared,ion_type=r.ion_type,pKa_true=r.pKa_true,macro=json.dumps(mp)))
    print(r.ID,mp,round(time.time()-t0),flush=True)
pd.DataFrame(rows).to_csv('/tmp/claude-0/pkrun/pkasso_macro_raw.csv',index=False)
