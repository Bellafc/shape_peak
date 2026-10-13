"""跨層遷移 steering：在第 X 層建方向、套到第 Y 層。python steer_xfer.py <repo或路徑> <concept> <名稱>（MK 指定家族）。輸出 results/steer_xfer/<名稱>_<concept>.csv（template, X, Y, hit, p_tgt, p_other, cos_last）"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
"""（原 steer_true 前置）真正的 steering：不借別題的激活，直接加「目標 centroid − 當前 centroid」。python steer_true.py <repo 或路徑> <concept> <名稱>（MK 環境變數指定家族）
- centroid 只用「其他兩個模板、且模型答對」的題計算（被 steer 的模板不參與）；每個位置、每層各自一組 centroid
- 只 steer 模型答對的題；目標 = (答案 + j) mod n，j 在題間輪流取 1…n−1
- 方式：加差值 α=1 / α=2；只加差值的二維平面成分；把 centroid 子空間成分換成目標 centroid；隨機方向（同範數）對照
- 位置：只改最後位置 / 問句後全部位置；每層記錄「輸出 = 目標」比例與目標機率"""
import sys, os, numpy as np, pandas as pd, torch
from huggingface_hub import snapshot_download
REPO, CON, NAME = sys.argv[1:4]
path = REPO if os.path.isdir(REPO) else snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"])
sys.argv = ["group.py", os.environ.get("MK", "qwen"), CON, "ans", path]
src_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_lib", "group.py")).read().split("# ---------- hook 與執行 ----------")[0]
src_ = src_.replace('O = os.environ.get("OUT") or f"{DATA}/group/{MNAME}/{CON}/{POS}"', 'O = f"{DATA}/steer_xfer"; os.makedirs(O, exist_ok=True)')
exec(src_)
keep = np.arange(N)  # 只改問句之後的位置（從結尾數，chat 模板結尾每題相同），所以不需要整題等長；前向時按長度分批
Pt = {}
for t in range(3):
    i = keep[tt[keep] == t][0]; toks = [tok.decode(x) for x in R.ids[i]]; Pt[t] = len(toks) - max(j for j, s in enumerate(toks) if "?" in s)
P = max(Pt.values())
st = {"L": None}
def hook(L):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1: return out
        if st.get("cap"): st["H"][L] = h[:, -P:].float().cpu()
        if st["L"] == L: h[:, -P:] = h[:, -P:] + st["delta"].to(h.dtype)
        return out
    return f
for L in range(NL): lm.layers[L].register_forward_hook(hook(L))
def fwd(idx, L=None, delta=None, cap=False):
    idx = list(idx); cs = torch.zeros(len(idx), n); os_ = torch.zeros(len(idx)); by = {}
    for j, v in enumerate(idx): by.setdefault(len(R.ids[v]), []).append(j)
    for _, js_all in by.items():
        for b in range(0, len(js_all), 32):
            js = js_all[b:b + 32]; ids = torch.tensor([R.ids[idx[j]] for j in js], device="cuda")
            st.update(L=L, delta=None if delta is None else delta[js].cuda(), cap=cap, H={})
            with torch.no_grad(): p = model(input_ids=ids, logits_to_keep=1).logits[:, -1].float().softmax(-1)
            st.update(L=None, cap=False); c = torch.zeros(len(js), n, device="cuda").index_add_(1, VC, p[:, VI]); cs[js] = c.cpu(); os_[js] = (1 - c.sum(1)).cpu()
            if cap:
                for L_ in range(NL): HT[[idx[j] for j in js], L_] = st["H"][L_]
    return cs, os_
HT = torch.zeros(N, NL, P, D); ok = np.zeros(N, bool)
for t in range(3):
    idx = list(keep[tt[keep] == t]); c, o = fwd(idx, cap=True); ok[idx] = torch.cat([c, o[:, None]], 1).argmax(1).numpy() == ans[idx]
import json; 

# ---------- 跨層遷移：第 X 層的 centroid 差值方向，按第 Y 層自己差值的範數縮放，加在第 Y 層問句後全部位置 ----------
EARLY = float(os.environ.get("XF_EARLY", 0.25)); LS = list(range(int(round(NL * EARLY)), NL))
# Y 只測「原生 steering（全部位置 α=1）首次 ≥ 0.3 的層 − 2」之後；更早的 Y 原生就無效，遷移不可能有效
_st = f"{DATA}/steer_true/{NAME}_{CON}.csv"; LY = LS
if os.path.exists(_st):
    _d = pd.read_csv(_st); _d = _d[(_d.positions == "問句後全部位置") & (_d.method == "加差值 α=1")].groupby("layer").hit.mean()
    _f = [L for L in _d.index if _d[L] >= 0.3]
    if _f: LY = [L for L in LS if L >= _f[0] - 2]
print("Y layers:", LY, flush=True)
rows = []
for t in range(3):
    test = [i for i in keep[tt[keep] == t] if ok[i]]; train = [i for i in keep if tt[i] != t and ok[i]]
    if not test: continue
    have = set(ans[train].tolist()); jj = np.array([1 + (r % (n - 1)) for r in range(len(test))]); a_ = ans[test]; b_ = (a_ + jj) % n
    sel = np.array([(x in have) and (y in have) for x, y in zip(a_, b_)]); test = [i for i, s_ in zip(test, sel) if s_]; a_, b_ = a_[sel], b_[sel]
    if not test: continue
    Pq = Pt[t]; mask = torch.zeros(P); mask[P - Pq:] = 1; mask = mask[None, :, None]; ytr = ans[train]
    DIF = {}
    for L in LS:
        Htr = HT[train, L]; C = torch.stack([Htr[ytr == k].mean(0) if (ytr == k).any() else torch.zeros(P, D) for k in range(n)])
        DIF[L] = (C[b_] - C[a_]) * mask  # T × P × D
    for Y in LY:
        nat = DIF[Y].norm(dim=-1, keepdim=True)
        for X in LS:
            v = DIF[X] / DIF[X].norm(dim=-1, keepdim=True).clamp(min=1e-6) * nat
            cosv = float(torch.nn.functional.cosine_similarity(DIF[X][:, -1], DIF[Y][:, -1], dim=-1).mean())
            c, o = fwd(test, Y, v)
            hit = torch.cat([c, o[:, None]], 1).argmax(1).numpy() == b_
            rows.append(dict(template=t, X=X, Y=Y, n=len(test), hit=float(hit.mean()), p_tgt=float(c[torch.arange(len(test)), torch.tensor(b_)].mean()),
                             p_other=float(o.mean()), cos_last=cosv))
        print("t", t, "Y", Y, flush=True)
        pd.DataFrame(rows).to_csv(f"{O}/{NAME}_{CON}.csv", index=False)
print("done", flush=True)
