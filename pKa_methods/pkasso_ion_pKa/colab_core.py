import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
D=pd.read_csv('drugs.csv'); I=pd.read_csv('ions.csv'); P=pd.read_csv('positives.csv')
dcols=[c for c in D.columns if c.startswith('d') and c[1:].isdigit()]; icols=[c for c in I.columns if c.startswith('i') and c[1:].isdigit()]
dsign=D.sign.values; isign=I.sign.values; nd,ni=len(D),len(I)
pos=[set() for _ in range(nd)]
for a,b in P.values:
    if isign[b]==-dsign[a]: pos[a].add(b)
cand=[np.where(isign==-dsign[i])[0] for i in range(nd)]
dp=D.pKa.values.astype(float); ip=I.pKa.values.astype(float); perm=I.perm.values
def clean(A): return np.clip(np.nan_to_num(A.astype(float)),-1e5,1e5)
def make_features(train_ids, ncomp=25):
    """PCA fitted on training drugs / all ions (no labels used) -> compact numeric features."""
    sd=StandardScaler().fit(clean(D.loc[train_ids,dcols].values)); pd_=PCA(ncomp,random_state=0).fit(sd.transform(clean(D.loc[train_ids,dcols].values)))
    si=StandardScaler().fit(clean(I[icols].values)); pi=PCA(ncomp,random_state=0).fit(si.transform(clean(I[icols].values)))
    Zd=pd_.transform(sd.transform(clean(D[dcols].values))); Zi=pi.transform(si.transform(clean(I[icols].values)))
    return Zd,Zi
def rows(i,js,Zd,Zi):
    n=len(js); dpk=(dp[i]-ip[js]) if dsign[i]>0 else (ip[js]-dp[i])
    return np.c_[np.tile(Zd[i],(n,1)),Zi[js],dpk,np.full(n,dp[i]),ip[js],perm[js]]
def evaluate(make_model, seeds=(0,1), max_train_rows=8000, verbose=True):
    use=[i for i in range(nd) if pos[i]]; out=[]
    for seed in seeds:
        rng=np.random.RandomState(seed); folds=np.array_split(rng.permutation(use),5)
        for te in folds:
            tr=[i for i in use if i not in set(te)]; Zd,Zi=make_features(tr)
            X=[];y=[]
            for i in tr:
                js=cand[i]; X.append(rows(i,js,Zd,Zi)); y.append(np.isin(js,list(pos[i])).astype(int))
            X=np.vstack(X); y=np.concatenate(y)
            pidx=np.where(y==1)[0]; nidx=np.where(y==0)[0]
            nidx=rng.choice(nidx,min(len(nidx),max_train_rows-len(pidx)),replace=False); sel=np.r_[pidx,nidx]
            m=make_model(); m.fit(X[sel],y[sel])
            cnt=np.zeros(ni)
            for i in tr:
                for j in pos[i]: cnt[j]+=1
            for i in te:
                js=cand[i]; sc_m=np.full(ni,-1e9); sc_m[js]=m.predict_proba(rows(i,js,Zd,Zi))[:,1]
                for nm,sc in [('model',sc_m),('popularity (train drugs)',cnt+1e-6*rng.rand(ni)),('random',rng.rand(ni))]:
                    ranks=[]
                    for p in pos[i]:
                        oth=[q for q in pos[i] if q!=p]; jj=np.array([j for j in js if j not in oth])
                        ranks.append((p,1+(sc[jj]>sc[p]).sum()+0.5*((sc[jj]==sc[p]).sum()-1)))
                    best=min(r for _,r in ranks)
                    out.append(dict(method=nm,seed=seed,drug=i,hit10=best<=10,mrr=1/best,rare=[r for p,r in ranks if cnt[p]<=2]))
        if verbose: print('seed',seed,'done',flush=True)
    R=pd.DataFrame(out)
    per=R.groupby(['method','drug'])[['hit10','mrr']].mean().reset_index().groupby('method')[['hit10','mrr']].mean().round(3)
    rr=pd.DataFrame([(m,r) for m,l in zip(R.method,R.rare) for r in l],columns=['method','rank'])
    rare=rr.groupby('method').agg(n=('rank','size'),hit10=('rank',lambda s:(s<=10).mean()),mrr=('rank',lambda s:(1/s).mean()),median_rank=('rank','median')).round(3)
    return per,rare
