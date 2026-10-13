import pandas as pd, numpy as np, ast
R=pd.read_csv('conv_layers.csv'); R['top']=R.top.apply(ast.literal_eval)
out=[]; P=lambda *a:(print(*a),out.append(' '.join(map(str,a))))
P('# 理論②（卷積模型）驗證：起點+k 任務\n')
P('share_pred = 答案 centroid 能量中「加性部分經 k 卷積」的比例；share_int = 「按答案組織的交互項」比例（兩者加上交叉項 = 1）。')
P('交叉層 = 晚期（≥25% 深度）第一個 share_int > share_pred 並維持到最後的層。\n')
P('| 模型 | 任務 | 第1峰 | 前三峰 | 交叉層 | 交叉層−最近的峰 | 峰前5層 share_pred | 峰後5層 share_int | 加性 R²（峰前5層） | 星期：峰前 cos(c, −a) / ‖c‖/‖a‖（預測 1 / 0.167） |')
P('|---|---|---|---|---|---|---|---|---|---|')
rows=[]
for (m,t),g in R.groupby(['model','task'],sort=False):
    g=g.sort_values('L'); p1=g.peak1.iloc[0]; top=g.top.iloc[0]; NL=g.NL.iloc[0]; lo=int(round(NL*0.25))
    dom=(g.share_int>g.share_pred).values; Ls=g.L.values
    cross=next((Ls[i] for i in range(len(Ls)) if Ls[i]>=lo and dom[i:].all()),np.nan)
    pre=g[(g.L>=p1-5)&(g.L<p1)]; post=g[(g.L>p1)&(g.L<=p1+5)]
    dnear=(cross-min(top,key=lambda x:abs(x-cross))) if np.isfinite(cross) and top else np.nan
    wk=f'{pre.cos_c_negA.mean():.2f} / {pre.ratio_c_A.mean():.3f}' if t=='weekday' else ''
    rows.append(dict(model=m,task=t,p1=p1,cross=cross,dnear=dnear,dp1=cross-p1 if p1>=0 else np.nan,pre_pred=pre.share_pred.mean(),post_int=post.share_int.mean()))
    P(f'| {m.replace("-Instruct","").replace("-v0.3","")} | {t} | {p1} | {top} | {cross} | {dnear:+.0f} | {pre.share_pred.mean():.2f} | {post.share_int.mean():.2f} | {pre.R2_add.mean():.2f} | {wk} |')
X=pd.DataFrame(rows); v=X.dropna(subset=['dnear'])
P(f'\n交叉層落在某個前三峰 ±1：{(v.dnear.abs()<=1).sum()}/{len(v)}；落在第1峰 ±1：{(v.dp1.abs()<=1).sum()}/{len(v)}；峰前 share_pred 中位 {X.pre_pred.median():.2f}，峰後 share_int 中位 {X.post_int.median():.2f}')
X.to_csv('conv_summary.csv',index=False); open('conv_check.md','w').write('\n'.join(out)+'\n')
