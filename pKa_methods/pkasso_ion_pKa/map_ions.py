import pandas as pd, numpy as np
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.MolStandardize import rdMolStandardize as S
U=S.Uncharger()
def neut(s):
    try: m=Chem.MolFromSmiles(s); Chem.RemoveStereochemistry(m); return Chem.MolToSmiles(U.uncharge(m))
    except Exception: return None
I=pd.read_csv('../../PPT_story_code/4_base_LightGBM/ions.csv')
v=pd.read_csv('../../IL_ion_dataset_579_v6_amino_acid_rule.csv')
I['key']=I.smiles.map(neut); v['key']=v.smiles.map(neut)
v['sg']=np.where(v.ion_type=='anion',-1,1); I['sg']=I.sign
vv=v.drop_duplicates(['key','sg']).set_index(['key','sg'])
k=list(zip(I.key,I.sg)); g=lambda c:[vv[c].get(x,np.nan) for x in k]
I['ion_id']=g('ion_id'); I['v6_pKa']=g('pKa'); I['v6_status']=g('pKa_status'); I['v6_conf']=g('pKa_confidence')
print('mapped',I.ion_id.notna().sum(),'of',len(I)); print(I.v6_status.value_counts(dropna=False).to_dict())
print('perm in ions.csv',int(I.perm.sum()),'| pKa NaN in ions.csv',int(I.pKa.isna().sum()))
d=(I.pKa-I.v6_pKa).abs(); print('ions.csv pKa differs from v6 by >0.01:',int((d>0.01).sum()))
print(I[d>0.01][['ion_id','name','pKa','v6_pKa','v6_status']].head(20).to_string())
I.to_csv('ions_mapped_to_v6.csv',index=False)
