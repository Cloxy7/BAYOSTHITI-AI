# Published pretrained pKa model MolGpKa (Pan et al., JCIM 2021, site-level GNN) on our measured counter-ions.
import sys, os, pickle, warnings, numpy as np, pandas as pd; warnings.filterwarnings('ignore')
sys.path.insert(0,'/tmp/claude-0/pkaml/molgpka/src'); os.chdir('/tmp/claude-0/pkaml/molgpka/src')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import predict_pka as P
OUT='../../pKa_more_methods/'
d=pd.read_csv('../../IL_ion_dataset_579_v5_pKa_fixed.csv'); conf=d.pKa_confidence.fillna('').str.lower()
known=d[(d.pKa_status=='measured')|((d.pKa_status=='web lookup')&conf.str.startswith('measured |'))].copy()
ours=pickle.load(open('/tmp/claude-0/pkaml/our_prepared_df.pkl','rb')); in138=set(ours.index)
rows=[]
for r in known.itertuples():
    m=Chem.MolFromSmiles(r.smiles)
    if m is None: continue
    try: b,a=P.predict(m)
    except Exception as e: b,a={},{}
    av=list(a.values()); bv=list(b.values())
    pred=(min(av) if av else np.nan) if r.ion_type=='anion' else (max(bv) if bv else np.nan)
    rows.append(dict(ion_id=r.ion_id,name=r.name,ion_type=r.ion_type,pKa_true=r.pKa,molgpka_pKa=float(pred) if pred==pred else np.nan,n_acid_sites=len(av),n_base_sites=len(bv),in_138_set=r.ion_id in in138))
R=pd.DataFrame(rows); R.to_csv(OUT+'molgpka_predictions_on_our_ions.csv',index=False)
def sc(t,p): e=p-t; return dict(n=len(t),MAE=np.abs(e).mean(),RMSE=np.sqrt((e**2).mean()),R2=1-(e**2).sum()/((t-t.mean())**2).sum())
ok=R.dropna(subset=['molgpka_pKa']); print('predicted for',len(ok),'of',len(R))
for lab,sub in [('all measured ions with a prediction',ok),('same 138 ions as the RF test',ok[ok.in_138_set]),('anions',ok[ok.ion_type=='anion']),('cations',ok[ok.ion_type=='cation']),('true pKa 2-12',ok[(ok.pKa_true>=2)&(ok.pKa_true<=12)])]:
    print(lab,{k:round(v,3) for k,v in sc(sub.pKa_true.values,sub.molgpka_pKa.values).items()})
