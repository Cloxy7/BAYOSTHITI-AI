# Base LightGBM recommender, same leave-drug-out test (5 folds x 2 seeds), with 4 versions of the ION pKa. Drug pKa is never changed.
import numpy as np, pandas as pd, lightgbm as lgb, warnings, shutil; warnings.filterwarnings('ignore')
I0=pd.read_csv('ions_mapped_to_v6.csv'); K=pd.read_csv('pkasso_310_ions.csv').set_index('row')
I0['pkasso']=K.pkasso_pKa.reindex(I0.index).values
# status: measured = 'measured' or web lookup whose confidence starts with 'measured |'
conf=I0.v6_conf.fillna('').str.lower()
I0['is_measured']=(I0.v6_status=='measured')|((I0.v6_status=='web lookup')&conf.str.startswith('measured |'))
FIXIDS=['AN_N047','CAT_N045','AN_N046','AN_N048','CAT_040','AN_151','AN_161','CAT_N002']
fix=I0.pKa.copy(); m=I0.ion_id.isin(FIXIDS); fix[m]=I0.v6_pKa[m]
A=fix.copy(); mA=(~I0.is_measured)&(I0.perm==0)&I0.pkasso.notna(); A[mA]=I0.pkasso[mA]
B=fix.copy(); mB=(I0.perm==0)&I0.pkasso.notna(); B[mB]=I0.pkasso[mB]
V={'current (as in 0.769 run)':I0.pKa,'amino-acid fix only':fix,'A: fix + pKasso for non-measured ions':A,'B: pKasso for every ion it can predict':B}
print('ions changed by fix:',int(m.sum()),'| A replaces',int(mA.sum()),'non-measured ions | B replaces',int(mB.sum()))
print('ions with no pKa: current',int(I0.pKa.isna().sum()),'| A',int(A.isna().sum()),'| B',int(B.isna().sum()))
def make(): return lgb.LGBMClassifier(n_estimators=200,learning_rate=0.05,num_leaves=15,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=0)
res={}; summ=[]
for nm,p in V.items():
    I=I0.drop(columns=['pkasso']).copy(); I['pKa']=p.values; I.drop(columns=[c for c in I.columns if c in('key','sg','ion_id','v6_pKa','v6_status','v6_conf','is_measured')]).to_csv('ions.csv',index=False)
    ns={}; exec(open('colab_core.py').read().replace("return per,rare","return per,rare,R"),ns)
    per,rare,R=ns['evaluate'](make,seeds=(0,1),verbose=False); res[nm]=R
    summ.append(dict(version=nm,hit10=per.loc['model','hit10'],mrr=per.loc['model','mrr'],pop_hit10=per.loc['popularity (train drugs)','hit10'],rare_hit10=rare.loc['model','hit10']))
    print(summ[-1],flush=True)
S=pd.DataFrame(summ); rng=np.random.RandomState(0)
h={k:R[R.method=='model'].groupby('drug').hit10.mean() for k,R in res.items()}
base=h['current (as in 0.769 run)']; pop=res['current (as in 0.769 run)'].pipe(lambda R:R[R.method=='popularity (train drugs)'].groupby('drug').hit10.mean())
for k in h:
    d=h[k]-base; bs=[d.sample(len(d),replace=True,random_state=rng).mean() for _ in range(2000)]
    e=h[k]-pop; bs2=[e.sample(len(e),replace=True,random_state=rng).mean() for _ in range(2000)]
    S.loc[S.version==k,'change_vs_current']=round(d.mean(),3); S.loc[S.version==k,'CI_lo']=round(np.percentile(bs,2.5),3); S.loc[S.version==k,'CI_hi']=round(np.percentile(bs,97.5),3)
    S.loc[S.version==k,'gap_vs_popularity']=round(e.mean(),3); S.loc[S.version==k,'gap_CI_lo']=round(np.percentile(bs2,2.5),3); S.loc[S.version==k,'gap_CI_hi']=round(np.percentile(bs2,97.5),3)
print(S.to_string(index=False)); S.to_csv('recommender_ion_pKa_versions_results.csv',index=False)
out=I0[['ion_id','name','smiles','sign','perm','is_measured','v6_status']].copy()
for k,p in V.items(): out[k]=p.values
out['pkasso_raw']=I0.pkasso; out.to_csv('ion_pKa_all_versions.csv',index=False)
