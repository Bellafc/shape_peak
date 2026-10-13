"""交接測試：輸入 token steering vs 答案位置 steering，逐層。python steer_handoff.py <repo 或路徑> <concept> <名稱>（MK 環境變數指定家族）
- 只用模型答對的題（第一個 token 合併機率判定）。每題挑新起點 s' = (s + j) mod n（j 在題間輪流取 1…n−1），目標 b = (s' + k) mod n。
- 輸入 token：在起點詞的最後一個 token（R.cpos）加「新起點 centroid − 原起點 centroid」（按起點分組，在該 token 該層算）。
- 答案位置：問句之後的全部位置（ans_all）或只改最後位置（ans_last），加「目標答案 centroid − 原答案 centroid」（按答案分組，逐位置逐層）。
- centroid 只用「其他兩個問法、且模型答對」的題（被 steer 的問法不參與）。
- 強度：nat（α = 1，不縮放）、rel0.5 / rel1.0（把向量縮放成 ‖Δ‖ = r·‖h‖，逐位置）。對照：同範數（rel1.0）的隨機方向（rand_in、rand_ans）。
- 每層記錄：輸出 = 目標 b 的比例（hit）、維持原答案的比例（keep）、「其他」機率（other）。
輸出 data/handoff/<名稱>_<concept>.csv"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, os, numpy as np, pandas as pd, torch
from huggingface_hub import snapshot_download
REPO, CON, NAME = sys.argv[1:4]
path = REPO if os.path.isdir(REPO) else snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"])
sys.argv = ["group.py", os.environ.get("MK", "qwen"), CON, "ans", path]
src_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_lib", "group.py")).read().split("# ---------- hook 與執行 ----------")[0]
src_ = src_.replace('O = os.environ.get("OUT") or f"{DATA}/group/{MNAME}/{CON}/{POS}"; os.makedirs(O, exist_ok=True)', 'O = f"{DATA}/handoff"; os.makedirs(O, exist_ok=True)')
exec(src_)
assert not NEWCON, "只支援起點 + k 的任務"
Pt = {}
for t in range(3):
    i = np.where(tt == t)[0][0]; toks = [tok.decode(x) for x in R.ids[i]]; Pt[t] = len(toks) - max(j for j, s in enumerate(toks) if "?" in s)
P = max(Pt.values()); assert all(len(R.ids[i]) - R.cpos[i] > P for i in range(N)), "起點詞落在問句後的位置"
st = {"L": None, "cap": False}
def hook(L):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1: return out
        if st["cap"]: st["HC"][L] = h[torch.arange(h.shape[0]), st["cpos"]].float().cpu(); st["HP"][L] = h[:, -P:].float().cpu()
        if st["L"] == L:
            d = st["delta"].to(h.dtype)
            if st["where"] == "in": h[torch.arange(h.shape[0]), st["cpos"]] += d
            else: h[:, -P:] += d
        return out
    return f
for L in range(NL): lm.layers[L].register_forward_hook(hook(L))
by = {}
for i in range(N): by.setdefault((tt[i], len(R.ids[i])), []).append(i)
def fwd(idx, L=None, where=None, delta=None, cap=False):
    """idx 任意；按（問法, 長度）分批。delta：(len(idx), D)（in）或 (len(idx), P, D)（ans）"""
    idx = list(idx); pos = {v: j for j, v in enumerate(idx)}; cp = torch.zeros(len(idx), n); oth = torch.zeros(len(idx)); g = {}
    for v in idx: g.setdefault((tt[v], len(R.ids[v])), []).append(v)
    HC = HP = None
    if cap: HC = torch.zeros(len(idx), NL, D, dtype=torch.float16); HP = torch.zeros(len(idx), NL, P, D, dtype=torch.float16)
    for _, vs in g.items():
        for b in range(0, len(vs), 32):
            sub = vs[b:b + 32]; rows_ = [pos[v] for v in sub]
            st.update(L=L, where=where, cap=cap, HC={}, HP={}, cpos=torch.tensor([R.cpos[v] for v in sub], device="cuda"),
                      delta=None if delta is None else delta[rows_].cuda())
            with torch.no_grad(): p = model(input_ids=torch.tensor([R.ids[v] for v in sub], device="cuda"), logits_to_keep=1).logits[:, -1].float().softmax(-1)
            c = torch.zeros(len(sub), n, device="cuda").index_add_(1, VC, p[:, VI]); cp[rows_] = c.cpu(); oth[rows_] = (1 - c.sum(1)).cpu()
            if cap:
                for L_ in range(NL): HC[rows_, L_] = st["HC"][L_].half(); HP[rows_, L_] = st["HP"][L_].half()
            st.update(L=None, cap=False)
    return cp, oth, HC, HP
cp0, o0, HC, HP = fwd(range(N), cap=True)
ok = torch.cat([cp0, o0[:, None]], 1).argmax(1).numpy() == ans
print(f"{NAME} {CON}：答對 {ok.sum()} / {N}", flush=True)
test = np.where(ok)[0]; jj = np.array([1 + (r % (n - 1)) for r in range(len(test))])
s_new = (src[test] + jj) % n; tgt = (s_new + kk[test]) % n
# 只留下所需 centroid 都存在的題（其他兩個問法裡，新起點 / 原起點 / 目標答案 / 原答案都要有答對的題）；同 steer_true.py 的做法。準確率高的組不受影響
def _has(t, a, by): return ((tt != t) & ok & (by == a)).any()
keep_ = np.array([_has(tt[i], s_new[q], src) and _has(tt[i], src[i], src) and _has(tt[i], tgt[q], ans) and _has(tt[i], ans[i], ans) for q, i in enumerate(test)])
if not keep_.all(): print(f"缺 centroid、略去 {(~keep_).sum()} 題", flush=True)
test, jj, s_new, tgt = test[keep_], jj[keep_], s_new[keep_], tgt[keep_]
mask = torch.zeros(len(test), P, 1)
for q, i in enumerate(test): mask[q, P - Pt[tt[i]]:] = 1  # 每個問法的問句後位置數不同
mlast = torch.zeros(len(test), P, 1); mlast[:, -1] = 1
gen = torch.Generator().manual_seed(0); rows = []
def scale(d, h, r):  # 逐向量縮放成 ‖d‖ = r‖h‖
    return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6) * (r * h.norm(dim=-1, keepdim=True))
for L in range(NL):
    hc, hp = HC[:, L].float(), HP[:, L].float()
    dIn = torch.zeros(len(test), D); dAns = torch.zeros(len(test), P, D)
    for t in range(3):
        tr = (tt != t) & ok; q_ = np.where(tt[test] == t)[0]
        Cs = {a: hc[tr & (src == a)].mean(0) for a in range(n) if (tr & (src == a)).any()}
        Ca = {a: hp[tr & (ans == a)].mean(0) for a in range(n) if (tr & (ans == a)).any()}
        for q in q_:
            i = test[q]; dIn[q] = Cs[s_new[q]] - Cs[src[i]]; dAns[q] = Ca[tgt[q]] - Ca[ans[i]]
    hIn = hc[test]; hAns = hp[test]
    rIn = torch.randn(dIn.shape, generator=gen); rAns = torch.randn(dAns.shape, generator=gen)
    conds = {"in_nat": ("in", dIn), "in_rel0.5": ("in", scale(dIn, hIn, 0.5)), "in_rel1.0": ("in", scale(dIn, hIn, 1.0)), "rand_in": ("in", scale(rIn, hIn, 1.0)),
             "ans_all_nat": ("ans", dAns * mask), "ans_all_rel0.5": ("ans", scale(dAns, hAns, 0.5) * mask), "ans_all_rel1.0": ("ans", scale(dAns, hAns, 1.0) * mask),
             "rand_ans": ("ans", scale(rAns, hAns, 1.0) * mask),
             "ans_last_nat": ("ans", dAns * mlast), "ans_last_rel0.5": ("ans", scale(dAns, hAns, 0.5) * mlast), "ans_last_rel1.0": ("ans", scale(dAns, hAns, 1.0) * mlast)}
    for nm, (where, d) in conds.items():
        cp, oth, _, _ = fwd(test, L, where, d); pr = torch.cat([cp, oth[:, None]], 1).argmax(1).numpy()
        rel = float((d.norm(dim=-1) / (hIn if where == "in" else hAns).norm(dim=-1).clamp(min=1e-6))[(d.norm(dim=-1) > 0)].mean())
        rows.append(dict(layer=L, cond=nm, hit=float((pr == tgt).mean()), keep=float((pr == ans[test]).mean()), other=float(oth.mean()), rel=rel, n=len(test)))
    pd.DataFrame(rows).to_csv(f"{O}/{NAME}_{CON}.csv", index=False)
    print("L", L, {r["cond"]: round(r["hit"], 2) for r in rows[-len(conds):] if r["cond"] in ("in_nat", "in_rel1.0", "ans_all_nat", "ans_all_rel1.0")}, flush=True)
print("done", flush=True)
