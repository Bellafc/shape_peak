"""強度掃描：形變峰是不是 steering 起效層的下界（橫線）。python steer_sweep.py <repo或路徑> <concept> <名稱>（MK 指定家族）
前半段（任務、centroid、判定）與 steer_xfer.py 相同；後半段：
- Y：前 25% 深度之後，到 max(第 1 峰 + 3, 原生 α=1 首次 ≥ 0.5 + 1)
- 方向：X = Y（原生）、X = 最後一層、X = 既有 steer_xfer 的事後最佳 X
- 強度：nat（α × 第 Y 層自己差值的範數，α = 1/2/4/8）；rel（c × 該題該位置第 Y 層殘差範數，c = 0.1/0.2/0.4/0.8）
- 對照：同範數隨機方向（nat α=8、rel c=0.4/0.8）
輸出 results/steer_sweep/<名稱>_<concept>.csv（template, Y, X, mode, s, n, hit, p_tgt, p_other, rel_nat）；rel_nat = 第 Y 層 ‖差值‖/‖殘差‖（最後位置平均），用來換算 α 與 c"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, os, numpy as np, pandas as pd, torch
from huggingface_hub import snapshot_download
REPO, CON, NAME = sys.argv[1:4]
path = REPO if os.path.isdir(REPO) else snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"])
sys.argv = ["group.py", os.environ.get("MK", "qwen"), CON, "ans", path]
src_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_lib", "group.py")).read().split("# ---------- hook 與執行 ----------")[0]
src_ = src_.replace('O = os.environ.get("OUT") or f"{DATA}/group/{MNAME}/{CON}/{POS}"', 'O = f"{DATA}/steer_sweep"; os.makedirs(O, exist_ok=True)')
exec(src_)
keep = np.arange(N)
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

# ---------- 強度掃描 ----------
LS = list(range(int(round(NL * 0.25)), NL))
E = pd.read_csv(f"{DATA}/events/events.csv"); E = E[(E.model == NAME) & (E.task == CON)]
PK = int(E[E.peak == "第1峰"].layer.iloc[0])
S = pd.read_csv(f"{DATA}/events/signals.csv"); S = S[(S.model == NAME) & (S.task == CON)].set_index("layer").sa
on1 = next((int(l) for l, v in S.items() if l >= LS[0] and v >= 0.5), PK)
LY = [L for L in LS if L <= min(NL - 1, max(PK + 3, on1 + 1))]
XB = NL - 1; _x = f"{DATA}/steer_xfer/{NAME}_{CON}.csv"
if os.path.exists(_x):
    _d = pd.read_csv(_x); _g = _d.groupby(["X", "Y"]).apply(lambda g: (g.hit * g.n).sum() / g.n.sum()).unstack()
    _nat = pd.Series({y: _g.loc[y, y] for y in _g.columns if y in _g.index}); _good = [y for y in _nat.index if _nat[y] >= 0.9]
    if _good: XB = int(_g[_good].mean(1).idxmax())
XS = sorted({NL - 1, XB}); NATA = [1, 2, 4, 8]; RELC = [0.1, 0.2, 0.4, 0.8]
print(f"peak L{PK}；原生 α=1 起效 L{on1}；Y {LY[0]}–{LY[-1]}；X = Y、{XS}", flush=True)
gen = torch.Generator().manual_seed(0); rows = []
for t in range(3):
    test = [i for i in keep[tt[keep] == t] if ok[i]]; train = [i for i in keep if tt[i] != t and ok[i]]
    if not test: continue
    have = set(ans[train].tolist()); jj = np.array([1 + (r % (n - 1)) for r in range(len(test))]); a_ = ans[test]; b_ = (a_ + jj) % n
    sel = np.array([(x in have) and (y in have) for x, y in zip(a_, b_)]); test = [i for i, s_ in zip(test, sel) if s_]; a_, b_ = a_[sel], b_[sel]
    if not test: continue
    Pq = Pt[t]; mask = torch.zeros(P); mask[P - Pq:] = 1; mask = mask[None, :, None]; ytr = ans[train]
    DIF = {}
    for L in sorted(set(LY) | set(XS)):
        Htr = HT[train, L]; C = torch.stack([Htr[ytr == k].mean(0) if (ytr == k).any() else torch.zeros(P, D) for k in range(n)])
        DIF[L] = (C[b_] - C[a_]) * mask
    unit = lambda v: v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    for Y in LY:
        nat = DIF[Y].norm(dim=-1, keepdim=True); res = HT[test, Y].norm(dim=-1, keepdim=True) * mask
        rel_nat = float((nat[:, -1] / res[:, -1].clamp(min=1e-6)).mean())
        R_ = unit(torch.randn(DIF[Y].shape, generator=gen)) * mask
        jobs = []
        for X in [Y] + [x for x in XS if x != Y]:
            u = unit(DIF[X])
            jobs += [(X, "nat", a, u * nat * a) for a in NATA] + [(X, "rel", c, u * res * c) for c in RELC]
        jobs += [("rand", "nat", 8, R_ * nat * 8), ("rand", "rel", 0.4, R_ * res * 0.4), ("rand", "rel", 0.8, R_ * res * 0.8)]
        for X, mode, s, v in jobs:
            c, o = fwd(test, Y, v); hit = torch.cat([c, o[:, None]], 1).argmax(1).numpy() == b_
            rows.append(dict(template=t, Y=Y, X=X, mode=mode, s=s, n=len(test), hit=float(hit.mean()),
                             p_tgt=float(c[torch.arange(len(test)), torch.tensor(b_)].mean()), p_other=float(o.mean()), rel_nat=rel_nat))
        print("t", t, "Y", Y, flush=True)
        pd.DataFrame(rows).to_csv(f"{O}/{NAME}_{CON}.csv", index=False)
print("done", flush=True)
