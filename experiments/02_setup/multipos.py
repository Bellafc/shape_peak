"""多步寫入：在形變層把「所有帶答案的位置」一起改，是否一次就夠？python multipos.py <repo> <concept> <名稱>
位置組：{最後}、{最後 + 中繼}（Llama：<|start_header_id|>；Mistral：'?'）、{問句 '?' 起到最後的全部位置}
換法：整個殘差；二維平面（各位置各自的 PCA-2，按答案分組）；centroid 張成子空間（n−1 維，各位置各自）
每層、每組、每種換法：正規化 LD（合併答案 token 機率）與翻轉率。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, os, numpy as np, pandas as pd, torch
from huggingface_hub import snapshot_download
REPO, CON, NAME = sys.argv[1:4]
path = REPO if os.path.isdir(REPO) else snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"])
sys.argv = ["group.py", os.environ.get("MK", "qwen"), CON, "ans", path]
src_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_lib", "group.py")).read().split("# ---------- hook 與執行 ----------")[0]
src_ = src_.replace('O = os.environ.get("OUT") or f"{DATA}/group/{MNAME}/{CON}/{POS}"', 'O = f"{DATA}/multipos"')
exec(src_)
# 每個模板只用最常見長度的題（位置對齊）
keep = np.arange(N)  # 只改問句之後的位置（從結尾數，chat 模板結尾每題相同），所以不需要整題等長；前向時按長度分批
tail = {}  # 模板 → (P = 從 '?' 到結尾的 token 數, 中繼位置的「距結尾」偏移)
for t in range(3):
    i = keep[tt[keep] == t][0]; toks = [tok.decode(x) for x in R.ids[i]]; Lq = len(toks)
    q = max(j for j, s in enumerate(toks) if "?" in s)
    relay = [j for j, s in enumerate(toks) if "start_header" in s]
    relay = relay[-1] if relay else q
    tail[t] = (Lq - q, Lq - relay, toks[q:])
P = max(v[0] for v in tail.values())
st = {"L": None}
def hook(L):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1: return out
        if st.get("cap"): st["H"][L] = h[:, -P:].float().cpu()
        if st["L"] == L: h[:, -P:] = h[:, -P:] + st["delta"].to(h.dtype)   # delta：B×P×D（非選中位置為 0）
        return out
    return f
for L in range(NL): lm.layers[L].register_forward_hook(hook(L))
def fwd(idx, L=None, delta=None, cap=False):
    idx = list(idx); cs = torch.zeros(len(idx), n); os_ = torch.zeros(len(idx)); by = {}
    for j, v in enumerate(idx): by.setdefault(len(R.ids[v]), []).append(j)
    for _, js in by.items():
        ids = torch.tensor([R.ids[idx[j]] for j in js], device="cuda"); st.update(L=L, delta=None if delta is None else delta[js].cuda(), cap=cap, H={})
        with torch.no_grad(): p = model(input_ids=ids, logits_to_keep=1).logits[:, -1].float().softmax(-1)
        st.update(L=None, cap=False); c = torch.zeros(len(js), n, device="cuda").index_add_(1, VC, p[:, VI]); cs[js] = c.cpu(); os_[js] = (1 - c.sum(1)).cpu()
        if cap:
            for L_ in range(NL): HT[[idx[j] for j in js], L_] = st["H"][L_]
    return cs, os_
HT = torch.zeros(N, NL, P, D)
for t in range(3):
    idx = keep[tt[keep] == t]
    fwd(list(idx), cap=True)
LDf = lambda cp, tgt, base: ((cp + 1e-12).log()[torch.arange(len(cp)), list(tgt)] - (cp + 1e-12).log()[torch.arange(len(cp)), list(base)]).numpy()
rng = np.random.default_rng(0); rows = []
for t in range(3):
    idx = keep[tt[keep] == t]; pairs = []
    while len(pairs) < 40:
        a, b = rng.choice(idx, 2, replace=False)
        if ans[a] != ans[b] and src[a] != src[b] and kk[a] != kk[b]: pairs.append((a, b))
    A_, B_ = [p[0] for p in pairs], [p[1] for p in pairs]; tgt, base = ans[B_], ans[A_]
    cA, _ = fwd(A_); cB, _ = fwd(B_); lA, lB = LDf(cA, tgt, base), LDf(cB, tgt, base); ok_ = (lB - lA) > 1
    Pt, relay, toks = tail[t]
    SETS = {"只改最後位置": [P - 1], "最後 + 中繼": sorted({P - 1, P - relay}), "問句後全部位置": list(range(P - Pt, P))}
    for L in range(NL):
        Hl = HT[idx, L]; labs = ans[idx]
        bases = {}
        for j in range(P - Pt, P):
            C = torch.stack([Hl[labs == a_, j].mean(0) for a_ in range(n)]); m = C.mean(0); _, S, Vh = torch.linalg.svd(C - m, full_matrices=False)
            bases[j] = (Vh[:2].T, Vh[: n - 1].T)
        for sname, js in SETS.items():
            for how in ("整個殘差", "二維平面", "centroid 子空間"):
                dlt = torch.zeros(len(A_), P, D)
                for j in js:
                    d = HT[B_, L, j] - HT[A_, L, j]
                    if how == "二維平面": U = bases[j][0]; d = (d @ U) @ U.T
                    if how == "centroid 子空間": U = bases[j][1]; d = (d @ U) @ U.T
                    dlt[:, j] = d
                cp, po = fwd(A_, L, dlt)
                rows.append(dict(template=t, layer=L, positions=sname, how=how, norm=float(np.mean(((LDf(cp, tgt, base) - lA) / (lB - lA))[ok_])),
                                 flip=float(np.mean(torch.cat([cp, po[:, None]], 1).argmax(1).numpy() == tgt))))
    pd.DataFrame(rows).to_csv(f"{O}/{NAME}_{CON}.csv", index=False); print("template", t, toks, "relay offset", relay, flush=True)
print("done", flush=True)
