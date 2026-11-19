"""Cohen kappa and image-cluster bootstrap helpers for human ratings."""
import numpy as np

def matrix(x,y,k):
    x=np.asarray(x,dtype=int);y=np.asarray(y,dtype=int)
    assert len(x)==len(y) and len(x)>0
    assert ((x>=0)&(x<k)&(y>=0)&(y<k)).all()
    return np.bincount(x*k+y,minlength=k*k).reshape(k,k).astype(float)

def kappa(cm,weighted=False):
    n=cm.sum()
    if not n:return None
    expected=np.outer(cm.sum(axis=1),cm.sum(axis=0))/n
    k=len(cm)
    weights=np.abs(np.arange(k)[:,None]-np.arange(k)[None,:]) if weighted else (1-np.eye(k))
    denominator=(weights*expected).sum()
    return float(1-(weights*cm).sum()/denominator) if denominator>0 else None

def pair_result(items,k):
    cm=matrix([i[1] for i in items],[i[2] for i in items],k)
    files=sorted({i[0] for i in items})
    blocks=np.array([matrix([i[1] for i in items if i[0]==fn],[i[2] for i in items if i[0]==fn],k) for fn in files])
    rng=np.random.default_rng(20261117);boot=[[],[]]
    for _ in range(1000):
        sample=blocks[rng.integers(0,len(blocks),size=len(blocks))].sum(axis=0)
        for j,w in enumerate([False,True]):
            v=kappa(sample,w)
            if v is not None:boot[j].append(v)
    ci=lambda a:list(map(float,np.quantile(a,[.025,.975]))) if len(a)>=800 else [None,None]
    return {'caption_pairs':len(items),'unique_images':len(files),'exact_agreement':float(np.trace(cm)/cm.sum()),'unweighted_kappa':kappa(cm),'unweighted_95pct_ci':ci(boot[0]),'linear_weighted_kappa':kappa(cm,True),'linear_weighted_95pct_ci':ci(boot[1]),'confusion_matrix':cm.astype(int).tolist(),'rater1_counts':cm.sum(axis=1).astype(int).tolist(),'rater2_counts':cm.sum(axis=0).astype(int).tolist()}

