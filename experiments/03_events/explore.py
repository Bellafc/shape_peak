import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import pandas as pd, numpy as np
E=pd.read_csv(f'{DATA}/events/events.csv')
S=pd.read_csv(f'{DATA}/events/signals.csv')
late=E[E.peak.isin(['第1峰','第2–3峰'])]
rows=[]
for r in late.itertuples():
    s=S[(S.model==r.model)&(S.task==r.task)].set_index('layer').sort_index()
    L=r.layer; NL=r.NL
    g=lambda k,l: s[k].get(l,np.nan)
    lens=s.lens; fis=s.fisher; idp=s.id
    lon=lens[lens>=0.1].index.min()  # lens onset
    fmax=fis.max()
    rows.append(dict(model=r.model[:6],task=r.task,L=L,pk=r.peak,type=r.type,depth=round(L/NL,2),
      lensL=round(g('lens',L),2),lens_m2=round(g('lens',L-2),2),lens_p3=round(g('lens',min(L+3,NL-1)),2),
      lon_rel=(lon-L) if pd.notna(lon) else np.nan, idL=round(g('id',L),2),id_m2=round(g('id',L-2),2),
      fisL=round(g('fisher',L)/fmax,2) if fmax==fmax else np.nan, fis_m2=round(g('fisher',L-2)/fmax,2) if fmax==fmax else np.nan,
      ratio=r.ratio_to_median))
X=pd.DataFrame(rows); X.to_csv('features.csv',index=False)
pd.set_option('display.width',250)
for t,g in X.groupby('type'): print('==',t,len(g)); print(g.drop(columns='type').to_string(index=False))
