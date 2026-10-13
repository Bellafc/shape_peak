"""區分事件種類：每組（模型 × 任務）列出所有形變小峰，對照同一層附近 patching / steering / lens / 身份探針的跳升，給每個事件分類。
信號（全部在答案位置，層號 = 第 L 個 decoder 層的輸出）：
  shape  全維 Procrustes 層間變化（per_layer.csv change_full）
  id     身份探針；pl 只改最後位置整個殘差 patching（per_layer full_norm）
  pa / pm / pc  multipos：問句後全部位置整個殘差 / 最後 + 中繼整個殘差 / 全部位置 centroid 子空間
  sa / sl  steer_true：全部位置 / 只改最後，加差值 α=1 的成功率
  lens   正確答案的 lens 機率；fisher（有存激活或 new_metrics 的組）
輸出：results/events/{signals.csv, events.csv, jumps.csv, summary.md, grid.png}"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import os, glob, json, numpy as np, pandas as pd, torch
R = f"{DATA}"; OUT = f"{R}/events"; os.makedirs(OUT, exist_ok=True)
JUMP = 0.10  # 機率型信號一層內上升 ≥ 0.10 算跳升
EARLY = 0.25  # 前 25% 深度的形變（嵌入 / 模板處理）另列，不參與分類

def mp_series(m, t, positions, how):
    f = f"{R}/multipos/{m}_{t}.csv"
    if not os.path.exists(f): return None
    d = pd.read_csv(f); d = d[(d.positions == positions) & (d.how == how)]
    return d.groupby("layer").norm.mean() if len(d) else None
def st_series(m, t, positions):
    f = f"{R}/steer_true/{m}_{t}.csv"
    if not os.path.exists(f): return None
    d = pd.read_csv(f); d = d[(d.positions == positions) & (d.method == "加差值 α=1")]
    return d.groupby("layer").hit.mean() if len(d) else None
def fisher_from_acts(f):
    a = torch.load(f, weights_only=False); H, y = a["H"].float(), np.asarray(a["ans"])
    out = []
    for L in range(H.shape[1]):
        X = H[:, L].numpy().astype(np.float64); C = np.stack([X[y == c].mean(0) for c in np.unique(y)]); mu = X.mean(0)
        within = np.mean([((X[y == c] - C[i]) ** 2).sum(1).mean() for i, c in enumerate(np.unique(y))])
        out.append(((C - mu) ** 2).sum(1).mean() / within)
    return pd.Series(out)

groups = []
for f in sorted(glob.glob(f"{R}/group/*/*/ans/per_layer.csv")):
    m, t = f.split("/")[-4:-2]; d = pd.read_csv(f).set_index("layer")
    groups.append((m, t, d.change_full, d.identity, d.full_norm))
for t, sh, ca in (("weekday", "weekday_add/shape.csv", "weekday_add/causal.csv"), ("month", "weekday_add/months/ans.csv", "weekday_add/months/ans.csv")):
    if not os.path.exists(f"{R}/{sh}"): continue  # 原 Qwen2.5-7B 的補充結果；只跑新模型時沒有
    s, c = pd.read_csv(f"{R}/{sh}").set_index("layer"), pd.read_csv(f"{R}/{ca}").set_index("layer")
    groups.append(("Qwen2.5-7B-Instruct", t, s.change_full, pd.Series(np.nan, index=s.index), c.full_norm))

rows = []
for m, t, shape, ident, pl in groups:
    S = pd.DataFrame({"shape": shape, "id": ident, "pl": pl})
    for k, ser in (("pa", mp_series(m, t, "問句後全部位置", "整個殘差")), ("pm", mp_series(m, t, "最後 + 中繼", "整個殘差")),
                   ("pc", mp_series(m, t, "問句後全部位置", "centroid 子空間")),
                   ("sa", st_series(m, t, "問句後全部位置")), ("sl", st_series(m, t, "只改最後位置"))):
        S[k] = ser if ser is not None else np.nan
    lf = f"{R}/lens/{m}_{t}.csv"; S["lens"] = pd.read_csv(lf).set_index("layer").lens_p if os.path.exists(lf) else np.nan
    nm = f"{R}/new_metrics/{m}_{t}/new_metrics.csv"; ac = f"{R}/acts/{m}_{t}_acts.pt"
    S["fisher"] = pd.read_csv(nm).set_index("layer").fisher if os.path.exists(nm) else (fisher_from_acts(ac) if os.path.exists(ac) else np.nan)
    S["model"], S["task"] = m, t
    rows.append(S.reset_index())
sig = pd.concat(rows); sig.to_csv(f"{OUT}/signals.csv", index=False)

CAUSAL = ["pl", "pa", "pm", "sa", "sl"]
def jumps(S, k):
    x = S[k].values; return [int(S.layer.iloc[i]) for i in range(1, len(x)) if np.isfinite(x[i]) and np.isfinite(x[i - 1]) and x[i] - x[i - 1] >= JUMP]
ev, jl = [], []
for (m, t), S in sig.groupby(["model", "task"], sort=False):
    S = S.sort_values("layer").reset_index(drop=True); NL = len(S); x = S["shape"].values; lo = int(round(NL * EARLY))
    late = x[lo:]; med = np.nanmedian(late)
    J = {k: jumps(S, k) for k in CAUSAL + ["pc", "id", "lens"]}
    J["id"] = [L for L in J["id"] if S.id[L] - S.id[L - 1] >= 0.15]
    for k, Ls in J.items():
        for L in Ls: jl.append(dict(model=m, task=t, signal=k, layer=L, delta=S[k][L] - S[k][L - 1]))
    peaks = [L for L in range(1, NL) if np.isfinite(x[L]) and x[L] >= np.nanmax(x[max(1, L - 1):L + 2]) and x[L] >= 1.5 * med]
    rank = {L: r + 1 for r, L in enumerate(sorted(range(lo, NL), key=lambda L: -np.nan_to_num(x[L])))}
    prank = {L: r + 1 for r, L in enumerate(sorted([p for p in peaks if p >= lo], key=lambda L: -x[L]))}
    near = lambda k, L, w=1: [j for j in J[k] if abs(j - L) <= w]
    val = lambda k, l: S[k][l] if 0 <= l < NL else np.nan
    has_c = S[CAUSAL].notna().any().any()
    for L in sorted(set(range(lo, NL)) | set(peaks)):
        cj = sorted({k for k in CAUSAL if near(k, L)}); pc_j = bool(near("pc", L)); id_j = bool(near("id", L)); lens_j = bool(near("lens", L))
        causal_now = np.nanmax([val(k, min(L + 1, NL - 1)) for k in CAUSAL]) if has_c else np.nan
        pa_before = np.nanmax([val("pa", L - 2), val("pl", L - 2), -1])
        if L < lo: typ = "早期"
        elif pc_j and pa_before >= 0.6: typ = "集中"
        elif cj: typ = "寫入"
        elif (id_j or val("id", L) >= 0.8) and causal_now < 0.2: typ = "可讀重組"
        elif lens_j: typ = "輸出化"
        else: typ = "無事件"
        pr = prank.get(L)
        ev.append(dict(model=m, task=t, NL=NL, layer=L, depth=round(L / NL, 2), shape=round(x[L], 4), shape_rank=rank.get(L), ratio_to_median=round(x[L] / med, 1),
                       peak="早期峰" if (L in peaks and L < lo) else ("第1峰" if pr == 1 else ("第2–3峰" if pr in (2, 3) else ("其他小峰" if pr else "非峰"))),
                       type=typ, causal_jumps=",".join(cj), pc_jump=pc_j, id_jump=id_j, lens_jump=lens_j,
                       id=round(val("id", L), 2), pl=round(val("pl", L), 2), pa=round(val("pa", L), 2), lens=round(val("lens", L), 3),
                       causal_next=round(causal_now, 2) if np.isfinite(causal_now) else np.nan))
E = pd.DataFrame(ev); E.to_csv(f"{OUT}/events.csv", index=False); pd.DataFrame(jl).to_csv(f"{OUT}/jumps.csv", index=False)
late = E[E.peak != "早期峰"]
tab = pd.crosstab(late.peak, late.type, normalize="index").round(2); tab["n"] = late.peak.value_counts()
print(tab.loc[["第1峰", "第2–3峰", "其他小峰", "非峰"]].to_string())
# 每種因果 / lens 跳升，落在第 1–3 峰 ±1 層內的比例；對照：隨機層落在峰 ±1 內的比例
J_ = pd.read_csv(f"{OUT}/jumps.csv"); top = late[late.peak.isin(["第1峰", "第2–3峰"])]
def near_top(r): p = top[(top.model == r.model) & (top.task == r.task)].layer.values; return bool(len(p)) and np.min(np.abs(p - r.layer)) <= 1
J_["near_peak"] = [near_top(r) for r in J_.itertuples()]
base = np.mean([near_top(r) for r in late.itertuples()])
print("各信號跳升落在第 1–3 峰 ±1 層內的比例（隨機晚期層的基準 %.2f）" % base)
print(J_.groupby("signal").near_peak.agg(["mean", "size"]).round(2).to_string())
