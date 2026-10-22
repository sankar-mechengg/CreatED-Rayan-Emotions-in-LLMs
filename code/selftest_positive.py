"""Positive-control check of analyse_experiments.py: synthetic data with injected effects for all confirmatory hypotheses."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_paths import (
    DATA_CSV, EARLIER_OVERLAP, CACHE, RESULTS, STIMULI, REFERENCE, FIGURES,
    PIPELINE, load_nrc_lexicon, text_open, apply_figure_style, ensure_runtime_dirs,
)
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(STIMULI))
ensure_runtime_dirs()

import numpy as np, analyse_experiments as A
f1,f2,f3,f4,J=A.synthetic()
EM=A.EM
# inject: D increases signed axis; evidence shifts A; target emotion raised in F3; express raises gen_t
def inj1(r):
    s=list(r.stim); a,b=r.pair.split("-"); sgn=1 if r.gold==a else -1
    if r.cond=="DISCRIMINATE": s[EM.index(a)]+=0.5*sgn
    lp=list(r.lp); lp[EM.index(r.gold)]+= s[EM.index(a)]*sgn
    r["stim"]=s; r["final"]=s; r["lp"]=lp; return r
f1=f1.apply(inj1,axis=1)
def inj2(r):
    s=list(r.final); d={"EV_A":1,"EV_B":-1}.get(r.cond,0); s[EM.index(r.a)]+=1.5*d
    lp=list(r.lp); lp[EM.index(r.a)]+=s[EM.index(r.a)]; r["final"]=s; r["lp"]=lp; return r
f2=f2.apply(inj2,axis=1)
labs=["anger","disgust","fear","joy","sadness","surprise","neutral"]
J=np.full((6,6),0.02); 
for a,b,v in [(0,1,.3),(2,4,.25),(3,5,.2),(0,2,.1),(1,4,.08)]: J[a,b]=J[b,a]=v
np.fill_diagonal(J,1)
rng=np.random.default_rng(5)
def inj3(r):
    t=EM.index(r.target); s=np.array(r.gen_score); s[t]+=1.5; s+=3*np.where(np.arange(6)==t,0,J[t])
    r["gen_score"]=s.tolist()
    m=s[t]-np.delete(s,t).mean()
    p=np.exp(np.r_[s*1.5, 0]); p/=p.sum()
    r["clf"]=dict(zip(labs,p.tolist())); return r
f3=f3.apply(inj3,axis=1)
def inj4(r):
    s=list(r.gen_score)
    if r.cond=="EXPRESS": s[EM.index(r.target)]+=1
    r["gen_score"]=s; d=dict(r.clf); k={"ANGER":"anger","FEAR":"fear"}[r.target]
    d[k]=float(1/(1+np.exp(-s[EM.index(r.target)]-rng.normal(0,1.5)))); r["clf"]=d; return r
f4=f4.apply(inj4,axis=1)
A.synthetic=lambda:(f1,f2,f3,f4,J)
A.main(selftest=True)
