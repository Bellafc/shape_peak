"""既有數據（不重跑模型）：原生 steering（sa / sl / α=2）、patching（pa / pl）、跨層最佳 X（Y*）的起效層 vs 形變第 1 峰。python existing_onset.py；TH=閾值（預設 0.5）。輸出 results/steer_sweep/existing_onset.csv"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import pandas as pd, numpy as np, os
R=f"{DATA}"; TH=float(os.environ.get("TH",0.5))
E=pd.read_csv(f"{R}/events/events.csv"); S=pd.read_csv(f"{R}/events/signals.csv")
M=["Gemma-2-9B-it","Llama-3.1-8B-Instruct","Mistral-7B-Instruct-v0.3","Qwen2.5-7B-Instruct"]
CAT={"weekday":"計算","month":"計算","hour":"計算","letter":"計算","city":"檢索","continent":"檢索","capital2hop":"多跳","color":"顏色","ctxcolor":"顏色","emotion":"情緒"}
wm=lambda g: (g.hit*g.n).sum()/g.n.sum() if g.n.sum() else np.nan
rows=[]
for m in M:
  for t in CAT:
    p=E[(E.model==m)&(E.task==t)&(E.peak=="第1峰")]
    if p.empty: continue
    pk=int(p.layer.iloc[0]); NL=int(p.NL.iloc[0]); lo=int(np.ceil(0.25*NL)); ty=p.type.iloc[0]
    s=S[(S.model==m)&(S.task==t)].set_index("layer").sort_index()
    on=lambda col: next((int(l) for l,v in s[col].items() if l>=lo and v>=TH),None)
    r=dict(model=m.split("-")[0],task=t,cat=CAT[t],type=ty,NL=NL,peak=pk,sa=on("sa"),sl=on("sl"),pa=on("pa"),pl=on("pl"))
    f=f"{R}/steer_true/{m}_{t}.csv"
    if os.path.exists(f):
      d=pd.read_csv(f); d=d[(d.positions=="問句後全部位置")&(d.method=="加差值 α=2")]
      a2=d.groupby("layer").apply(wm); r["a2"]=next((int(l) for l,v in a2.items() if l>=lo and v>=TH),None)
      r["a2_prepeak_max"]=round(a2[(a2.index>=lo)&(a2.index<pk-1)].max(),2) if ((a2.index>=lo)&(a2.index<pk-1)).any() else np.nan
    f=f"{R}/steer_xfer/{m}_{t}.csv"
    if os.path.exists(f):
      x=pd.read_csv(f); g=x.groupby(["Y","X"]).apply(wm).groupby(level=0).max()
      r["Ymin"]=int(g.index.min()); r["Ystar"]=next((int(l) for l,v in g.items() if v>=TH),None)
      r["censored"]=r["Ystar"] is not None and r["Ystar"]<=r["Ymin"]
      r["n_test"]=int(x[(x.X==x.Y)].groupby("template").n.first().sum())
    r["sa_prepeak_max"]=round(s.sa[(s.index>=lo)&(s.index<pk-1)].max(),2)
    r["pa_prepeak_max"]=round(s.pa[(s.index>=lo)&(s.index<pk-1)].max(),2)
    rows.append(r)
D=pd.DataFrame(rows)
for c in ["sa","sl","pa","pl","a2","Ystar"]: D["d_"+c]=D[c]-D.peak
pd.set_option("display.width",300); pd.set_option("display.max_columns",40)
print(D.drop(columns=[c for c in D if c.startswith("d_")]).to_string(index=False))
D.to_csv(os.environ.get("OUT",f"{R}/steer_sweep/existing_onset.csv"),index=False)
