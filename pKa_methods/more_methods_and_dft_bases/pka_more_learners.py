# Try other learners for pKa on the same data as "Machine learning meets pKa" (Baltruschat & Czodrowski 2020):
# train on their ~5,900 literature pKa values; test on Novartis, AvLiLuMoVe and on our own 138 measured counter-ions.
import sys, pickle, warnings, numpy as np, pandas as pd, time; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger, DataStructs; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import PandasTools, AllChem, Descriptors, MACCSkeys, rdFingerprintGenerator
REPO='/tmp/claude-0/pkaml/repo'; sys.path.insert(0,REPO)
tr=PandasTools.LoadSDF(REPO+'/datasets/combined_training_datasets_unique_no_oe.sdf'); tr=tr[tr.ROMol.notna()].reset_index(drop=True); ytr=tr.pKa.astype(float).values
ours=pickle.load(open('/tmp/claude-0/pkaml/our_prepared_df.pkl','rb'))
can=lambda m:Chem.MolToSmiles(m)
overlap=set(map(can,tr.ROMol))&set(map(can,ours.ROMol)); keep=np.array([can(m) not in overlap for m in tr.ROMol])
print('training molecules',len(tr),'| removed because identical to one of our test ions:',int((~keep).sum()))
tr=tr[keep].reset_index(drop=True); ytr=ytr[keep]
def tests():
    out={}
    for nm,f in [('Novartis','novartis_cleaned_mono_unique_notraindata.sdf'),('AvLiLuMoVe','AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf')]:
        e=PandasTools.LoadSDF(REPO+'/datasets/'+f); e=e[e.ROMol.notna()]; out[nm]=(list(e.ROMol),e.pKa.astype(float).values)
    out['our measured ions']=(list(ours.ROMol),ours.pKa_true.astype(float).values)
    return out
T=tests()
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
def f_fcfp6(m): return np.array(AllChem.GetMorganFingerprintAsBitVect(m,3,4096,useFeatures=True),dtype=np.uint8)
names=[d[0] for d in Descriptors._descList]
def f_rich(m):
    cnt=gen.GetCountFingerprintAsNumPy(m).astype(float)
    mac=np.array(list(MACCSkeys.GenMACCSKeys(m)),dtype=float)
    ds=[]
    for n,fn in Descriptors._descList:
        try: ds.append(fn(m))
        except Exception: ds.append(np.nan)
    ds=np.clip(np.nan_to_num(np.array(ds,dtype=float),nan=0,posinf=0,neginf=0),-1e5,1e5)
    return np.r_[cnt,mac,ds]
FE={'FCFP6 bits (paper)':f_fcfp6,'rich: Morgan counts+MACCS+RDKit descriptors':f_rich}
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor
MODELS={
 'RandomForest (paper)':lambda:RandomForestRegressor(300,n_jobs=-1,random_state=24),
 'ExtraTrees':lambda:ExtraTreesRegressor(500,n_jobs=-1,random_state=24,min_samples_leaf=1),
 'XGBoost':lambda:xgb.XGBRegressor(n_estimators=600,learning_rate=0.04,max_depth=7,subsample=0.8,colsample_bytree=0.4,n_jobs=-1,random_state=24),
 'LightGBM':lambda:lgb.LGBMRegressor(n_estimators=800,learning_rate=0.03,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.3,verbose=-1,random_state=24),
 'CatBoost':lambda:CatBoostRegressor(iterations=1200,learning_rate=0.06,depth=8,verbose=0,random_seed=24,thread_count=-1),
 'SVR (RBF)':lambda:make_pipeline(StandardScaler(),SVR(C=3,epsilon=0.1,gamma='scale')),
 'MLP':lambda:make_pipeline(StandardScaler(),MLPRegressor((256,64),alpha=1e-1,max_iter=300,early_stopping=True,random_state=24)),
}
def sc(t,p): e=p-t; return dict(MAE=np.abs(e).mean(),RMSE=np.sqrt((e**2).mean()),R2=1-(e**2).sum()/((t-t.mean())**2).sum())
rows=[]; preds={}
for fn,f in FE.items():
    t0=time.time(); Xtr=np.array([f(m) for m in tr.ROMol]); Xte={k:np.array([f(m) for m in v[0]]) for k,v in T.items()}; print(fn,Xtr.shape,'features in %.0fs'%(time.time()-t0),flush=True)
    for mn,mk in MODELS.items():
        if mn=='SVR (RBF)' and fn.startswith('FCFP'): continue
        if mn=='MLP' and fn.startswith('FCFP'): continue
        t0=time.time(); m=mk().fit(Xtr,ytr)
        for k,(mols,y) in T.items():
            p=m.predict(Xte[k]); preds[(fn,mn,k)]=p; rows.append(dict(features=fn,model=mn,test_set=k,n=len(y),**sc(y,p)))
        print('  %-22s %.0fs  ours R2 %.3f MAE %.2f'%(mn,time.time()-t0,rows[-1]['R2'],rows[-1]['MAE']),flush=True)
R=pd.DataFrame(rows).round(3); R.to_csv('pka_more_learners_results.csv',index=False)
pickle.dump(preds,open('pka_more_learners_preds.pkl','wb'))
print(R[R.test_set=='our measured ions'].sort_values('MAE').to_string(index=False))
