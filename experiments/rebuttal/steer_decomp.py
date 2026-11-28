"""Rebuttal (rTuT C1): is the answer part of a centroid difference sufficient to steer, and the blurred-input part not?
Usage: MK=<family> python steer_decomp.py <repo or path> <weekday|month|hour|letter> <name>
Setup as in 04_handoff/steer_handoff.py (answer positions only): correctly answered prompts; each prompt gets a new start
s' = s + j and the target s' + k; centroids come from the other two templates (correct prompts only), at every position after
the question mark and every layer.
Decomposition (Sec. 7, App. J; as in 07_theory/joint_check.py): on those training prompts, at each layer and position, fit
h = mu + a(s) + b(k) + t(template) + u(D) by least squares. The answer centroid then splits into
B(D) = mean of a(s) over the prompts with answer D (blurred input) and U(D) = u(D) (answer).
Directions added at all answer positions (alpha = 1 unless noted):
  full       c(target) - c(original)            the paper's centroid difference
  answer     U(target) - U(original)
  blur       B(target) - B(original)
  answer_eq  answer, rescaled to the norm of full (per position)
  blur_eq    blur, rescaled to the norm of full
  random     random direction with the norm of full
Also logged per layer at the last position: share of ||U||^2 in ||B||^2 + ||U||^2 (no noise correction), and the cosine of full
with answer and with blur. Output: DATA/decomp/<name>_<task>.csv (layer, cond, hit, keep, other, n, ...)"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, os, numpy as np, pandas as pd, torch
REPO, CON, NAME = sys.argv[1:4]
path = REPO
sys.argv = ["group.py", os.environ.get("MK", "qwen"), CON, "ans", path]
src_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_lib", "group.py")).read().split("# ---------- hook 與執行 ----------")[0]
src_ = src_.replace('O = os.environ.get("OUT") or f"{DATA}/group/{MNAME}/{CON}/{POS}"; os.makedirs(O, exist_ok=True)', 'O = f"{DATA}/decomp"; os.makedirs(O, exist_ok=True)')
exec(src_)
assert not NEWCON, "cyclic-arithmetic tasks only"
Pt = {}
for t in range(3):
    i = np.where(tt == t)[0][0]; toks = [tok.decode(x) for x in R.ids[i]]; Pt[t] = len(toks) - max(j for j, s in enumerate(toks) if "?" in s)
P = max(Pt.values())
st = {"L": None, "cap": False}
def hook(L):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1: return out
        if st["cap"]: st["HP"][L] = h[:, -P:].float().cpu()
        if st["L"] == L: h[:, -P:] += st["delta"].to(h.dtype)
        return out
    return f
for L in range(NL): lm.layers[L].register_forward_hook(hook(L))
def fwd(idx, L=None, delta=None, cap=False):
    idx = list(idx); pos = {v: j for j, v in enumerate(idx)}; cp = torch.zeros(len(idx), n); oth = torch.zeros(len(idx)); g = {}
    for v in idx: g.setdefault((tt[v], len(R.ids[v])), []).append(v)
    HP = torch.zeros(len(idx), NL, P, D, dtype=torch.float16) if cap else None
    for _, vs in g.items():
        for b in range(0, len(vs), 32):
            sub = vs[b:b + 32]; rows_ = [pos[v] for v in sub]
            st.update(L=L, cap=cap, HP={}, delta=None if delta is None else delta[rows_].cuda())
            with torch.no_grad(): p = model(input_ids=torch.tensor([R.ids[v] for v in sub], device="cuda"), logits_to_keep=1).logits[:, -1].float().softmax(-1)
            c = torch.zeros(len(sub), n, device="cuda").index_add_(1, VC, p[:, VI]); cp[rows_] = c.cpu(); oth[rows_] = (1 - c.sum(1)).cpu()
            if cap:
                for L_ in range(NL): HP[rows_, L_] = st["HP"][L_].half()
            st.update(L=None, cap=False)
    return cp, oth, HP
cp0, o0, HP = fwd(range(N), cap=True)
ok = torch.cat([cp0, o0[:, None]], 1).argmax(1).numpy() == ans
print(f"{NAME} {CON}: correct {ok.sum()} / {N}", flush=True)
test = np.where(ok)[0]; jj = np.array([1 + (r % (n - 1)) for r in range(len(test))])
s_new = (src[test] + jj) % n; tgt = (s_new + kk[test]) % n
def _has(t, a): return ((tt != t) & ok & (ans == a)).any()
keep_ = np.array([_has(tt[i], tgt[q]) and _has(tt[i], ans[i]) for q, i in enumerate(test)])
if not keep_.all(): print(f"missing centroid, dropped {(~keep_).sum()}", flush=True)
test, s_new, tgt = test[keep_], s_new[keep_], tgt[keep_]
mask = torch.zeros(len(test), P, 1)
for q, i in enumerate(test): mask[q, P - Pt[tt[i]]:] = 1

def dummies(x, levels):
    M = np.zeros((len(x), len(levels))); M[np.arange(len(x)), np.searchsorted(levels, x)] = 1; return M[:, 1:]
def split(tr):
    """joint fit on training prompts tr; returns per-class centroid C, blurred part B and answer part U, each (n, P, D) for this layer"""
    s_, k_, t_, d_ = src[tr], kk[tr], tt[tr], ans[tr]
    Ls = [np.unique(x) for x in (s_, k_, t_, d_)]
    X = np.hstack([np.ones((len(tr), 1))] + [dummies(x, l) for x, l in zip((s_, k_, t_, d_), Ls)]); sz = [1] + [len(l) - 1 for l in Ls]
    return X, sz, Ls
gen = torch.Generator().manual_seed(0); rows = []
fits = {}
for t in range(3):
    tr = np.where((tt != t) & ok)[0]; X, sz, Ls = split(tr); fits[t] = (tr, torch.tensor(np.linalg.pinv(X), dtype=torch.float64), sz, Ls)
for L in range(NL):
    hp = HP[:, L].double()                           # (N, P, D)
    dF = torch.zeros(len(test), P, D); dU = torch.zeros_like(dF); dB = torch.zeros_like(dF); share, cF_U, cF_B = [], [], []
    for t in range(3):
        tr, Pinv, sz, Ls = fits[t]; q_ = np.where(tt[test] == t)[0]
        Y = hp[tr].reshape(len(tr), -1); mu = Y.mean(0); beta = Pinv @ (Y - mu)          # (cols, P*D)
        off = np.cumsum([0] + sz)
        def lev(j):                                    # levels of factor j (0 = s, 3 = D), baseline 0, centred over levels
            E = torch.cat([torch.zeros(1, beta.shape[1], dtype=beta.dtype), beta[off[j + 1]:off[j + 2]]]); return E - E.mean(0)
        a_, u_ = lev(0), lev(3); sl, dl = list(Ls[0]), list(Ls[3])
        C = {c: (Y[ans[tr] == c] - mu).mean(0) for c in dl}
        U = {c: u_[dl.index(c)] for c in dl}
        B = {c: a_[[sl.index(x) for x in src[tr][ans[tr] == c]]].mean(0) for c in dl}
        bu = sum(float((B[c].view(P, D)[-1] ** 2).sum()) for c in dl); uu = sum(float((U[c].view(P, D)[-1] ** 2).sum()) for c in dl)
        share.append(uu / (bu + uu + 1e-12))
        for q in q_:
            i = test[q]; a0, b0 = ans[i], tgt[q]
            dF[q] = (C[b0] - C[a0]).view(P, D).float(); dU[q] = (U[b0] - U[a0]).view(P, D).float(); dB[q] = (B[b0] - B[a0]).view(P, D).float()
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x[:, -1], y[:, -1], dim=-1).mean())
    nF = dF.norm(dim=-1, keepdim=True)
    eq = lambda d: d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6) * nF
    rnd = torch.randn(dF.shape, generator=gen)
    conds = {"full": dF, "answer": dU, "blur": dB, "answer_eq": eq(dU), "blur_eq": eq(dB), "random": eq(rnd)}
    info = dict(u_share=float(np.mean(share)), cos_full_answer=cos(dF, dU), cos_full_blur=cos(dF, dB),
                norm_answer=float((dU[:, -1].norm(dim=-1) / dF[:, -1].norm(dim=-1).clamp(min=1e-6)).mean()),
                norm_blur=float((dB[:, -1].norm(dim=-1) / dF[:, -1].norm(dim=-1).clamp(min=1e-6)).mean()))
    for nm, d in conds.items():
        cp, oth, _ = fwd(test, L, d * mask); pr = torch.cat([cp, oth[:, None]], 1).argmax(1).numpy()
        rows.append(dict(layer=L, cond=nm, hit=float((pr == tgt).mean()), keep=float((pr == ans[test]).mean()), other=float(oth.mean()), n=len(test), **info))
    pd.DataFrame(rows).to_csv(f"{O}/{NAME}_{CON}.csv", index=False)
    print("L", L, f"u_share {info['u_share']:.2f}", {r["cond"]: round(r["hit"], 2) for r in rows[-len(conds):]}, flush=True)
print("done", flush=True)
