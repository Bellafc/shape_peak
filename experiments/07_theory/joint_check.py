"""理論②的可識別版本（回應審稿：只擬合加性模型時，純答案表示 h = u(D) 也會被算成「模糊輸入」）。
聯合擬合 h = μ + a(s) + b(k) + t(模板) + u(D) + r（最小平方，每層）。K 含兩個以上 offset 時 a 與 u 可識別。
平衡設計下 E[b|D] = E[t|D] = E[r|D] = 0，所以答案 centroid 恰好 c(D) = B(D) + U(D)：
  B = E[a(s)|D] = (a * p_K)(D)（模糊輸入），U = u(D)（答案），‖c‖² = ‖B‖² + ‖U‖² + 2⟨B,U⟩。
‖B‖²、‖U‖² 扣掉噪聲的期望貢獻（σ² 由殘差估計）。輸出每層：blur = ‖B‖²/(‖B‖²+‖U‖²)、ans = 1 − blur、cross = 2⟨B,U⟩/‖c‖²、uniq_ans = 加入 u(D) 後多解釋的變異比例、
以及舊估計（只擬合加性模型）的 old_blur / old_int / R2_add 作對照。
用法：python joint_check.py <events.csv（含 peak 欄：第1峰 / 第2–3峰）> <acts 檔...>
acts 檔可以是 dims.py 的格式（H, ans, src, k, t）或 fail_pred 的格式（只有 H, ans；題目順序為 t × s × k，k = 1..6）。
另附合成資料檢查（--synthetic）：純答案、純輸入、混合，看估計能否還原。"""
import sys, os, numpy as np, pandas as pd, torch
NS = {'weekday': 7, 'month': 12, 'hour': 12, 'letter': 26}

def _dummies(x, n):
    M = np.zeros((len(x), n)); M[np.arange(len(x)), x] = 1; return M[:, 1:]

def _design(s, ki, ti, D, N, nk, nt, with_u=True):
    cols = [np.ones((len(s), 1)), _dummies(s, N), _dummies(ki, nk), _dummies(ti, nt)] + ([_dummies(D, N)] if with_u else [])
    return np.hstack(cols), [c.shape[1] for c in cols]

def _resid(X, Y): return Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]

def decompose(H, s, k, t, D, N):
    Hc = H - H.mean(0); tot = (Hc ** 2).sum()
    ks, ki = np.unique(k, return_inverse=True); ts, ti = np.unique(t, return_inverse=True)
    C = np.stack([Hc[D == i].mean(0) for i in range(N)]); EC = (C ** 2).sum()
    # 舊估計：平衡設計的主效應
    As = np.stack([Hc[s == i].mean(0) for i in range(N)]); Bk = np.stack([Hc[ki == j].mean(0) for j in range(len(ks))]); Ts = np.stack([Hc[ti == j].mean(0) for j in range(len(ts))])
    r = Hc - As[s] - Bk[ki] - Ts[ti]
    old_blur = (np.stack([As[s[D == i]].mean(0) for i in range(N)]) ** 2).sum() / EC
    old_int = (np.stack([r[D == i].mean(0) for i in range(N)]) ** 2).sum() / EC
    # 聯合估計
    X, sz = _design(s, ki, ti, D, N, len(ks), len(ts)); beta = np.linalg.lstsq(X, Hc, rcond=None)[0]
    lev = lambda j: (lambda E: E - E.mean(0))(np.vstack([np.zeros((1, Hc.shape[1])), beta[sum(sz[:j]):sum(sz[:j + 1])]]))
    a, u = lev(1), lev(4)
    B = np.stack([a[s[D == i]].mean(0) for i in range(N)])
    rf = Hc - X @ beta
    # 去偏：估計噪聲對 ‖B‖²、‖U‖² 的貢獻 σ²·‖M‖_F²，M = 由觀測到 B / U 的線性算子
    P = np.linalg.pinv(X); lv = lambda j: (lambda E: E - E.mean(0))(np.vstack([np.zeros((1, P.shape[1])), P[sum(sz[:j]):sum(sz[:j + 1])]]))
    Ma, Mu = lv(1), lv(4); Mb = np.stack([Ma[s[D == i]].mean(0) for i in range(N)])
    s2 = (rf ** 2).sum() / (len(Hc) - np.linalg.matrix_rank(X))
    eb, eu = max((B ** 2).sum() - s2 * (Mb ** 2).sum(), 0), max((u ** 2).sum() - s2 * (Mu ** 2).sum(), 0)
    X0, _ = _design(s, ki, ti, D, N, len(ks), len(ts), with_u=False)
    return dict(R2_add=1 - (r ** 2).sum() / tot, old_blur=old_blur, old_int=old_int,
                blur=eb / (eb + eu + 1e-12), ans=eu / (eb + eu + 1e-12), cross=2 * (B * u).sum() / EC,
                uniq_ans=((_resid(X0, Hc) ** 2).sum() - (rf ** 2).sum()) / tot)

def load(f, N):
    a = torch.load(f, weights_only=False); H = a['H'].float().numpy(); n = H.shape[0]
    if 'src' in a: s, k, t = (np.asarray(a[x]) for x in ('src', 'k', 't'))
    else:
        assert n == 3 * N * 6, (f, n); idx = np.arange(n); t = idx // (N * 6); s = (idx % (N * 6)) // 6; k = idx % 6 + 1
    D = np.asarray(a['ans']); assert ((s + k) % N == D).all(), f
    return H, s, k, t, D

def synthetic():
    rng = np.random.default_rng(0)
    for N in (7, 12, 26):
        s, k, t = (x.ravel() for x in np.meshgrid(np.arange(N), np.arange(1, 7), np.arange(3), indexing='ij')); D = (s + k) % N
        circ = np.zeros((len(D), 50)); circ[:, 0], circ[:, 1] = np.cos(2 * np.pi * D / N), np.sin(2 * np.pi * D / N)
        Ma, Mu = rng.normal(size=(N, 50)), rng.normal(size=(N, 50))
        for nm, H in [('answer only, circle', circ), ('answer only, random', Mu[D]), ('input only', Ma[s]), ('input + answer + noise', Ma[s] + 0.5 * Mu[D] + rng.normal(size=(len(D), 50)))]:
            r = decompose(H, s, k, t, D, N)
            print(f"N={N:2d} {nm:24s} additive-only: blurred {r['old_blur']:.2f}, answer {r['old_int']:.2f} | joint: blurred {r['blur']:.2f}, answer {r['ans']:.2f}")

if __name__ == '__main__':
    if sys.argv[1] == '--synthetic': synthetic(); sys.exit()
    E = pd.read_csv(sys.argv[1]); rows = []
    for f in sys.argv[2:]:
        m, t = os.path.basename(f)[:-8].rsplit('_', 1)
        if t not in NS: continue
        e = E[(E.model == m) & (E.task == t)]
        p1 = int(e[e.peak == '第1峰'].layer.iloc[0]) if len(e) else -1; top = sorted(e[e.peak.isin(['第1峰', '第2–3峰'])].layer.tolist())
        H, s, k, tt, D = load(f, NS[t])
        for L in range(H.shape[1]):
            rows.append(dict(model=m, task=t, L=L, NL=H.shape[1], peak1=p1, top=top, **decompose(H[:, L].astype(np.float64), s, k, tt, D, NS[t])))
        print(m, t, 'done', flush=True)
    R = pd.DataFrame(rows); R.to_csv('joint_layers.csv', index=False)
    out = []
    for (m, t), g in R.groupby(['model', 'task'], sort=False):
        g = g.sort_values('L'); p1, top, NL = g.peak1.iloc[0], g.top.iloc[0], g.NL.iloc[0]; lo = int(round(NL * 0.25))
        if p1 < 0: continue
        dom = (g.ans > g.blur).values; Ls = g.L.values
        cr = next((Ls[i] for i in range(len(Ls)) if Ls[i] >= lo and dom[i:].all()), np.nan)
        cro = next((Ls[i] for i in range(len(Ls)) if Ls[i] >= lo and (g.old_int > g.old_blur).values[i:].all()), np.nan)
        pre, post = g[(g.L >= p1 - 5) & (g.L < p1)], g[(g.L > p1) & (g.L <= p1 + 5)]
        near = lambda c: c - min(top, key=lambda x: abs(x - c)) if np.isfinite(c) and top else np.nan
        out.append(dict(model=m, task=t, p1=p1, top=top, cross=cr, dnear=near(cr), cross_old=cro, dnear_old=near(cro),
                        pre_blur=pre.blur.mean(), post_ans=post.ans.mean(), pre_cross=pre.cross.mean(), post_cross=post.cross.mean(),
                        pre_uniq_ans=pre.uniq_ans.mean(), post_uniq_ans=post.uniq_ans.mean(), pre_R2_add=pre.R2_add.mean(),
                        pre_old_blur=pre.old_blur.mean(), post_old_int=post.old_int.mean()))
    X = pd.DataFrame(out); X.to_csv('joint_summary.csv', index=False)
    v, vo = X.dropna(subset=['dnear']), X.dropna(subset=['dnear_old'])
    print(X.round(2).to_string())
    print(f"\njoint: pairs {len(X)}; crossings {len(v)}; within ±1 of a top-3 peak {(v.dnear.abs() <= 1).sum()}")
    print(f"       before peak: blurred {X.pre_blur.median():.2f}, unique answer variance {X.pre_uniq_ans.median():.4f}, cross term {X.pre_cross.median():+.2f}")
    print(f"       after peak:  answer {X.post_ans.median():.2f}, unique answer variance {X.post_uniq_ans.median():.3f}, cross term {X.post_cross.median():+.2f}")
    print(f"old:   crossings {len(vo)}; within ±1 {(vo.dnear_old.abs() <= 1).sum()}; pre blurred {X.pre_old_blur.median():.2f}, post answer {X.post_old_int.median():.2f}")
