"""事件分類 v2：因果信號改用交接實驗（同一批題、同一個目標、命中率），與正文圖 1、3 一致。
規則與 events_pack/overnight/event_types.py 相同，只換信號：
  pa = 問句後全部位置 patching 命中率、pl = 只換最後位置 patching、sa = 全部位置 steering（α=1）、sl = 只改最後位置 steering
  pc（centroid 子空間 patching，判「集中」用）沿用舊信號；id 沿用舊身份探針（Qwen 星期 / 月份從存下的激活補算）；lens = lens_full 的 lens_p。
參數化以便做門檻敏感度：JUMP（跳升門檻）、PEAKR（峰 ≥ 晚期中位數倍數）、WIN（±窗口）、EARLY（早期界線）。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import os, numpy as np, pandas as pd, torch, itertools
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
EV = f"{DATA}/events"; R = f"{DATA}"
S = pd.read_csv(f"{EV}/signals.csv")
BIG = ["Gemma-2-9B-it", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Phi-4-14B"]
ORIG4 = BIG[:4]  # 原論文 4 個模型的交接 / lens_full 原始數據已遺失，直接沿用原 signals_v2（signals_v2_orig4.csv）
NEW = BIG[4:]
STARTK = {"weekday", "month", "hour", "letter"}
def idprobe_from_acts(f):
    a = torch.load(f, weights_only=False); H = a["H"].float(); y = np.asarray(a["ans"]); out = []
    for L in range(H.shape[1]):
        X = H[:, L].numpy(); X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = PCA(min(64, X.shape[0] - 1), random_state=0).fit_transform(X)
        out.append(float(cross_val_score(LogisticRegression(max_iter=2000, C=0.1), X, y, cv=5).mean()))
    return pd.Series(out)
def build_signals():
    rows = [pd.read_csv(f"{R}/events_v2/signals_v2_orig4.csv")]
    for m in NEW:
        for t in sorted(S[S.model == m].task.unique()):
            if m.startswith("Mistral") and t == "hour": continue
            old = S[(S.model == m) & (S.task == t)].set_index("layer").sort_index()
            sd = "handoff" if t in STARTK else "handoff_retrieval"
            fp, fs = f"{R}/handoff_patch/{m}_{t}.csv", f"{R}/{sd}/{m}_{t}.csv"
            if not (os.path.exists(fp) and os.path.exists(fs)): print("缺交接數據", m, t); continue
            pa_, st_ = pd.read_csv(fp), pd.read_csv(fs); g = lambda d, c: d[d.cond == c].set_index("layer").hit
            lf = pd.read_csv(f"{R}/lens_full/{m}_{t}.csv").set_index("layer")
            ident = old["id"]  # 新模型的 group.py 都有身份探針，不需要補算
            if ident.isna().all():
                f = f"{DATA}/acts/{m}_{t}_acts.pt"; ident = idprobe_from_acts(f); print("補算身份探針", m, t)
            D = pd.DataFrame({"shape": old["shape"], "id": ident.values if len(ident) == len(old) else ident, "pa": g(pa_, "ans_all_patch"), "pl": g(pa_, "ans_last_patch"),
                              "sa": g(st_, "ans_all_nat"), "sl": g(st_, "ans_last_nat"), "pc": old["pc"], "lens": lf.lens_p, "a_lin": lf.a_lin})
            D["model"], D["task"] = m, t; rows.append(D.reset_index().rename(columns={"index": "layer"}))
    return pd.concat(rows)
CAUSAL = ["pa", "pl", "sa", "sl"]
def classify(sig, JUMP=0.10, PEAKR=1.5, WIN=1, EARLY=0.25):
    ev = []
    for (m, t), Sg in sig.groupby(["model", "task"], sort=False):
        Sg = Sg.sort_values("layer").reset_index(drop=True); NL = len(Sg); x = Sg["shape"].values; lo = int(round(NL * EARLY)); med = np.nanmedian(x[lo:])
        def jumps(k): v = Sg[k].values; return [i for i in range(1, NL) if np.isfinite(v[i]) and np.isfinite(v[i - 1]) and v[i] - v[i - 1] >= JUMP]
        J = {k: jumps(k) for k in CAUSAL + ["pc", "id", "lens"]}; J["id"] = [L for L in J["id"] if Sg.id[L] - Sg.id[L - 1] >= 0.15]
        peaks = [L for L in range(1, NL) if np.isfinite(x[L]) and x[L] >= np.nanmax(x[max(1, L - 1):L + 2]) and x[L] >= PEAKR * med]
        prank = {L: r + 1 for r, L in enumerate(sorted([p for p in peaks if p >= lo], key=lambda L: -x[L]))}
        near = lambda k, L: [j for j in J[k] if abs(j - L) <= WIN]
        val = lambda k, l: Sg[k][l] if 0 <= l < NL else np.nan
        for L in range(lo, NL):
            cj = sorted({k for k in CAUSAL if near(k, L)}); pc_j = bool(near("pc", L)); id_j = bool(near("id", L)); lens_j = bool(near("lens", L))
            causal_now = np.nanmax([val(k, min(L + 1, NL - 1)) for k in CAUSAL]); pa_before = np.nanmax([val("pa", L - 2), val("pl", L - 2), -1])
            if pc_j and pa_before >= 0.6: typ = "集中"
            elif cj: typ = "寫入"
            elif (id_j or val("id", L) >= 0.8) and causal_now < 0.2: typ = "可讀重組"
            elif lens_j: typ = "輸出化"
            else: typ = "無事件"
            pr = prank.get(L)
            ev.append(dict(model=m, task=t, NL=NL, layer=L, shape=x[L], peak="第1峰" if pr == 1 else ("第2–3峰" if pr in (2, 3) else ("其他小峰" if pr else "非峰")),
                           type=typ, causal_jumps=",".join(cj), id=val("id", L), pa=val("pa", L), sa=val("sa", L), lens=val("lens", L), a_lin=val("a_lin", L), causal_next=causal_now))
    return pd.DataFrame(ev)
if __name__ == "__main__":
    sig = build_signals(); sig.to_csv("signals_v2.csv", index=False)
    E2 = classify(sig); E2.to_csv("events_v2.csv", index=False); print(len(E2[E2.peak == "第1峰"]), "組")
