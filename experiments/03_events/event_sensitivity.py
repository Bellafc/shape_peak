"""事件分類的門檻敏感度：JUMP（跳升門檻）、PEAK（峰 ≥ 晚期中位數的倍數）、W（配對窗口 ±W 層）、EARLY（早期深度界線）。
每組設定重算：(a) 第 1 峰 / 非峰的類型分布；(b) 各信號跳升落在第 1–3 峰 ±W 內的比例 vs 隨機晚期層；(c) 可讀重組事件的 lens 與領先層數。
輸入 results/events/signals.csv；輸出 results/events/sensitivity.csv、sensitivity.md"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import itertools, numpy as np, pandas as pd
R = f"{DATA}/events"; sig = pd.read_csv(f"{R}/signals.csv")
FULL = {"Gemma-2-9B-it": 42, "Llama-3.1-8B-Instruct": 32, "Llama-3.2-3B-Instruct": 28, "Mistral-7B-Instruct-v0.3": 32,
        "Qwen2.5-1.5B-Instruct": 28, "Qwen2.5-3B-Instruct": 36, "Qwen2.5-7B-Instruct": 28, "Qwen3-4B": 36}
sig = sig[sig.groupby(["model", "task"]).layer.transform("size") == sig.model.map(FULL)]
CAUSAL = ["pl", "pa", "pm", "sa", "sl"]; SIGS = CAUSAL + ["pc", "id", "lens"]

def classify(JUMP, PEAK, W, EARLY):
    ev, jl = [], []
    for (m, t), S in sig.groupby(["model", "task"], sort=False):
        S = S.sort_values("layer").reset_index(drop=True); NL = len(S); x = S["shape"].values; lo = int(round(NL * EARLY)); med = np.nanmedian(x[lo:])
        J = {}
        for k in SIGS:
            v = S[k].values; th = max(JUMP, 0.15) if k == "id" else JUMP
            J[k] = [L for L in range(1, NL) if np.isfinite(v[L]) and np.isfinite(v[L - 1]) and v[L] - v[L - 1] >= th]
            jl += [dict(model=m, task=t, signal=k, layer=L) for L in J[k]]
        peaks = [L for L in range(lo, NL) if np.isfinite(x[L]) and x[L] >= np.nanmax(x[max(1, L - 1):L + 2]) and x[L] >= PEAK * med]
        prank = {L: r + 1 for r, L in enumerate(sorted(peaks, key=lambda L: -x[L]))}
        val = lambda k, l: S[k][l] if 0 <= l < NL else np.nan
        near = lambda k, L: any(abs(j - L) <= W for j in J[k])
        for L in range(lo, NL):
            cj = [k for k in CAUSAL if near(k, L)]
            cn = np.nanmax([val(k, min(L + 1, NL - 1)) for k in CAUSAL] + [-1])
            pb = np.nanmax([val("pa", L - 2), val("pl", L - 2), -1])
            if near("pc", L) and pb >= 0.6: typ = "集中"
            elif cj: typ = "寫入"
            elif (near("id", L) or val("id", L) >= 0.8) and cn < 0.2: typ = "可讀重組"
            elif near("lens", L): typ = "輸出化"
            else: typ = "無事件"
            pr = prank.get(L)
            ev.append(dict(model=m, task=t, layer=L, type=typ, peak="第1峰" if pr == 1 else ("第2–3峰" if pr in (2, 3) else ("其他小峰" if pr else "非峰")),
                           lens=val("lens", L)))
    E = pd.DataFrame(ev); Jd = pd.DataFrame(jl)
    top = E[E.peak.isin(["第1峰", "第2–3峰"])]; tl = top.groupby(["model", "task"]).layer.apply(np.array).to_dict()
    nearp = lambda m, t, L: (m, t) in tl and np.min(np.abs(tl[(m, t)] - L)) <= W
    Jd = Jd[Jd.layer >= Jd.apply(lambda r: int(round(FULL[r.model] * EARLY)), axis=1)]
    Jd["np"] = [nearp(r.model, r.task, r.layer) for r in Jd.itertuples()]
    base = np.mean([nearp(r.model, r.task, r.layer) for r in E.itertuples()])
    out = dict(JUMP=JUMP, PEAK=PEAK, W=W, EARLY=EARLY, n_top1=int((E.peak == "第1峰").sum()), base_near=round(base, 2))
    for p, nm in (("第1峰", "top1"), ("非峰", "non")):
        d = E[E.peak == p].type.value_counts(normalize=True)
        for ty, en in (("寫入", "write"), ("可讀重組", "reorg"), ("輸出化", "lensonly"), ("無事件", "none"), ("集中", "conc")): out[f"{nm}_{en}"] = round(d.get(ty, 0), 2)
    for k in ("id", "sa", "pl", "lens"): out[f"near_{k}"] = round(Jd[Jd.signal == k].np.mean(), 2)
    # 可讀重組（第 1–3 峰）：lens 是否為 0、因果 ≥0.3 / lens ≥0.1 晚幾層
    R_ = top[top.type == "可讀重組"]; lag_c, lag_l = [], []
    for r in R_.itertuples():
        S = sig[(sig.model == r.model) & (sig.task == r.task)].set_index("layer")
        c = S[CAUSAL].max(axis=1); lc = [L for L in c.index if L >= r.layer and c[L] >= 0.3]; ll = [L for L in S.index if L >= r.layer and S.lens[L] >= 0.1]
        if lc: lag_c.append(lc[0] - r.layer)
        if ll: lag_l.append(ll[0] - r.layer)
    out.update(n_reorg=len(R_), reorg_lens0=round((R_.lens < 0.01).mean(), 2) if len(R_) else np.nan,
               reorg_lag_causal=np.median(lag_c) if lag_c else np.nan, reorg_lag_lens=np.median(lag_l) if lag_l else np.nan)
    return out

rows = [classify(*p) for p in itertools.product([0.05, 0.10, 0.15, 0.20], [1.25, 1.5, 2.0, 3.0], [0, 1, 2], [0.20, 0.25, 0.33])]
D = pd.DataFrame(rows); D.to_csv(f"{R}/sensitivity.csv", index=False)
D["write_x"] = D.top1_write / D.non_write
D["id_minus_lens"] = D.near_id - D.near_lens
KEY = ["top1_write", "non_write", "write_x", "top1_lensonly", "top1_none", "non_none", "near_id", "near_sa", "near_pl", "near_lens", "base_near", "id_minus_lens",
       "n_reorg", "reorg_lens0", "reorg_lag_causal", "reorg_lag_lens"]
desc = D[KEY].describe(percentiles=[0.05, 0.5, 0.95]).T[["min", "5%", "50%", "95%", "max"]].round(2)
print(desc.to_string())
checks = {"第 1 峰「寫入」比例 > 非峰": (D.top1_write > D.non_write).mean(),
          "第 1 峰「只有 lens」= 0": (D.top1_lensonly == 0).mean(),
          "第 1 峰「無事件」< 非峰": (D.top1_none < D.non_none).mean(),
          "身份跳升靠近峰的比例 > lens": (D.near_id > D.near_lens).mean(),
          "steering 跳升靠近峰的比例 > lens": (D.near_sa > D.near_lens).mean(),
          "lens 跳升靠近峰的比例 − 基準 < 0.15": ((D.near_lens - D.base_near) < 0.15).mean(),
          "可讀重組全部 lens < 0.01": (D.reorg_lens0 == 1).mean()}
for k, v in checks.items(): print(f"{k}: {v:.2f}（{len(D)} 組設定中成立的比例）")
by = {p: D.groupby(p)[["top1_write", "write_x", "near_id", "near_sa", "near_lens", "base_near", "n_reorg"]].median().round(2) for p in ["JUMP", "PEAK", "W", "EARLY"]}
with open(f"{R}/sensitivity.md", "w") as f:
    f.write(f"# 門檻敏感度（{len(D)} 組設定：JUMP × PEAK × W × EARLY）\n\n## 各指標在所有設定下的分布\n\n" + desc.to_markdown() + "\n\n## 結論是否在每組設定下成立\n\n")
    f.write("\n".join(f"- {k}：{v:.2f}" for k, v in checks.items()) + "\n")
    for p, t in by.items(): f.write(f"\n## 按 {p}（其他參數取中位數）\n\n" + t.to_markdown() + "\n")
for p, t in by.items(): print(f"\n{p}\n{t.to_string()}")
