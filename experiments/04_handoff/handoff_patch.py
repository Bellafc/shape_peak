"""交接測試（重寫版；原 handoff_tentative_paper/patching/ 已遺失，按論文附錄 D / F 的描述重寫）：輸入 token vs 答案位置，逐層 patching；
檢索類任務另做 steering（原 handoff_retrieval/ 腳本也遺失；計算類的 steering 仍用 steer_handoff.py）。
python handoff_patch.py <repo 或路徑> <concept> <名稱>（MK 環境變數指定家族）
- 只用模型答對的題（第一個 token 合併機率判定）。目標：計算類每題新起點 s' = (s + j) mod n、目標 (s' + k) mod n；檢索類目標 = (答案 + j) mod n。j 在題間輪流取 1…n−1（同 steer_handoff.py）。
- patching 夥伴：同一問法、模型答對、且答案 = 目標的題；計算類另要求起點 = s'、k 相同；檢索類要求線索不同（多個候選時固定種子隨機取一個）。沒有夥伴的題不計（論文 n = 有配對夥伴的題數）。
- 輸入 token：起點詞 / 線索的最後一個 token（R.cpos）；上下文顏色是被問者綁定的顏色詞的最後一個 token；情緒是故事的最後一個 token。
- 答案位置：問句問號之後到結尾的全部位置（ans_all）或只有最後位置（ans_last）。patching = 把該層該位置的殘差整個換成夥伴同一位置（從結尾對齊）的殘差。
- 檢索類 steering（同 steer_handoff.py 的條件名）：輸入 token 加「目標答案 centroid − 原答案 centroid」（在線索 token、該層算，排除同一線索的題）；
  答案位置加「目標答案 centroid − 原答案 centroid」（逐位置）。centroid 只用其他兩個問法、模型答對的題。對照：同範數（rel1.0）的隨機方向。
輸出 data/handoff_patch/<名稱>_<concept>.csv（in_patch / ans_all_patch / ans_last_patch）；
檢索類另輸出 data/handoff_retrieval/<名稱>_<concept>.csv（in_nat / … / ans_last_nat，欄位同 steer_handoff.py）"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, os, re, numpy as np, pandas as pd, torch
from huggingface_hub import snapshot_download
REPO, CON, NAME = sys.argv[1:4]
path = REPO if os.path.isdir(REPO) else snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"])
sys.argv = ["group.py", os.environ.get("MK", "qwen"), CON, "ans", path]
src_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_lib", "group.py")).read().split("# ---------- hook 與執行 ----------")[0]
src_ = src_.replace('O = os.environ.get("OUT") or f"{DATA}/group/{MNAME}/{CON}/{POS}"; os.makedirs(O, exist_ok=True)', 'O = f"{DATA}/handoff_patch"; os.makedirs(O, exist_ok=True)')
exec(src_)
OS = f"{DATA}/handoff_retrieval"; os.makedirs(OS, exist_ok=True)
# ---------- 輸入 token 位置 ----------
cin = np.array(R.cpos.values)
if CON == "ctxcolor":  # 被問者綁定的顏色詞：「<名字> likes <顏色>.」裡的顏色詞
    for i in range(N):
        p = R.prompt[i]; who = re.search(r"What color does (\w+) like", p).group(1); m_ = re.search(rf"\b{who} likes (\w+)\.", p); a, b = m_.span(1)
        off = tok(p, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        cin[i] = [j for j, (x, y) in enumerate(off) if x < b and y > a][-1]
Pt = {}
for t in range(3):
    i = np.where(tt == t)[0][0]; toks = [tok.decode(x) for x in R.ids[i]]; Pt[t] = len(toks) - max(j for j, s in enumerate(toks) if "?" in s)
P = max(Pt.values()); assert all(len(R.ids[i]) - cin[i] > P for i in range(N)), "輸入 token 落在問句後的位置"
# ---------- hook：擷取 / patching（整個換掉）/ steering（加上去） ----------
st = {"L": None, "cap": False}
def hook(L):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1: return out
        ar = torch.arange(h.shape[0], device=h.device)
        if st["cap"]: st["HC"][L] = h[ar, st["cpos"]].float().cpu(); st["HP"][L] = h[:, -P:].float().cpu()
        if st["L"] == L:
            v = st["val"].to(h.dtype)
            if st["where"] == "in":
                h[ar, st["cpos"]] = v if st["mode"] == "patch" else h[ar, st["cpos"]] + v
            else:
                m_ = st["mask"].to(h.dtype)  # (B, P, 1)：只動問句後的位置
                h[:, -P:] = h[:, -P:] * (1 - m_) + v * m_ if st["mode"] == "patch" else h[:, -P:] + v * m_
        return out
    return f
for L in range(NL): lm.layers[L].register_forward_hook(hook(L))
def fwd(idx, L=None, where=None, val=None, mask=None, mode="patch", cap=False):
    """idx 任意；按（問法, 長度）分批。val：(len(idx), D)（in）或 (len(idx), P, D)（ans）；mask：(len(idx), P, 1)"""
    idx = list(idx); pos = {v: j for j, v in enumerate(idx)}; cp = torch.zeros(len(idx), n); oth = torch.zeros(len(idx)); g = {}
    for v in idx: g.setdefault((tt[v], len(R.ids[v])), []).append(v)
    HC = HP = None
    if cap: HC = torch.zeros(len(idx), NL, D); HP = torch.zeros(len(idx), NL, P, D)
    for _, vs in g.items():
        for b in range(0, len(vs), 32):
            sub = vs[b:b + 32]; rows_ = [pos[v] for v in sub]
            st.update(L=L, where=where, mode=mode, cap=cap, HC={}, HP={}, cpos=torch.tensor([cin[v] for v in sub], device="cuda"),
                      val=None if val is None else val[rows_].cuda(), mask=None if mask is None else mask[rows_].cuda())
            with torch.no_grad(): p = model(input_ids=torch.tensor([R.ids[v] for v in sub], device="cuda"), logits_to_keep=1).logits[:, -1].float().softmax(-1)
            c = torch.zeros(len(sub), n, device="cuda").index_add_(1, VC, p[:, VI]); cp[rows_] = c.cpu(); oth[rows_] = (1 - c.sum(1)).cpu()
            if cap:
                for L_ in range(NL): HC[rows_, L_] = st["HC"][L_]; HP[rows_, L_] = st["HP"][L_]
            st.update(L=None, cap=False)
    return cp, oth, HC, HP
cp0, o0, HC, HP = fwd(range(N), cap=True)
ok = torch.cat([cp0, o0[:, None]], 1).argmax(1).numpy() == ans
print(f"{NAME} {CON}：答對 {ok.sum()} / {N}", flush=True)
test = np.where(ok)[0]; jj = np.array([1 + (r % (n - 1)) for r in range(len(test))])
if NEWCON: tgt = (ans[test] + jj) % n
else: s_new = (src[test] + jj) % n; tgt = (s_new + kk[test]) % n
mask = torch.zeros(len(test), P, 1)
for q, i in enumerate(test): mask[q, P - Pt[tt[i]]:] = 1
mlast = torch.zeros(len(test), P, 1); mlast[:, -1] = 1
# ---------- patching 夥伴 ----------
rng = np.random.default_rng(0); partner = np.full(len(test), -1)
for q, i in enumerate(test):
    if NEWCON: c_ = np.where(ok & (tt == tt[i]) & (ans == tgt[q]) & (src != src[i]))[0]
    else: c_ = np.where(ok & (tt == tt[i]) & (src == s_new[q]) & (kk == kk[i]))[0]
    if len(c_): partner[q] = c_[0] if len(c_) == 1 else rng.choice(c_)
pq = np.where(partner >= 0)[0]; ptest, ppart, ptgt = test[pq], partner[pq], tgt[pq]
print(f"有夥伴 {len(pq)} / {len(test)}", flush=True)
# ---------- 逐層 ----------
gen = torch.Generator().manual_seed(0); rows, srows = [], []
def scale(d, h, r): return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6) * (r * h.norm(dim=-1, keepdim=True))
def rec(lst, L, nm, cp, oth, tg, base, extra={}):
    pr = torch.cat([cp, oth[:, None]], 1).argmax(1).numpy()
    lst.append(dict(layer=L, cond=nm, hit=float((pr == tg).mean()), keep=float((pr == base).mean()), other=float(oth.mean()), n=len(tg), **extra))
for L in range(NL):
    # patching
    if len(pq):
        for nm, where, val, mk in (("in_patch", "in", HC[ppart, L], None), ("ans_all_patch", "ans", HP[ppart, L], mask[pq]), ("ans_last_patch", "ans", HP[ppart, L], mlast[pq])):
            cp, oth, _, _ = fwd(ptest, L, where, val, mk, "patch"); rec(rows, L, nm, cp, oth, ptgt, ans[ptest])
        pd.DataFrame(rows).to_csv(f"{O}/{NAME}_{CON}.csv", index=False)
    # 檢索類 steering
    if NEWCON:
        hc, hp = HC[:, L], HP[:, L]; dIn = torch.zeros(len(test), D); dAns = torch.zeros(len(test), P, D)
        for t in range(3):
            tr = (tt != t) & ok
            Ca = {a: hp[tr & (ans == a)].mean(0) for a in range(n) if (tr & (ans == a)).any()}
            for q in np.where(tt[test] == t)[0]:
                i = test[q]; tr_c = tr & (src != src[i])  # 輸入端排除同一線索
                ca = lambda a: hc[tr_c & (ans == a)].mean(0) if (tr_c & (ans == a)).any() else None
                a0, a1 = ca(ans[i]), ca(tgt[q])
                if a0 is not None and a1 is not None: dIn[q] = a1 - a0
                if tgt[q] in Ca and ans[i] in Ca: dAns[q] = Ca[tgt[q]] - Ca[ans[i]]
        hIn = hc[test]; hAns = hp[test]; rIn = torch.randn(dIn.shape, generator=gen); rAns = torch.randn(dAns.shape, generator=gen)
        conds = {"in_nat": ("in", dIn, None), "in_rel0.5": ("in", scale(dIn, hIn, 0.5), None), "in_rel1.0": ("in", scale(dIn, hIn, 1.0), None), "rand_in": ("in", scale(rIn, hIn, 1.0), None),
                 "ans_all_nat": ("ans", dAns, mask), "ans_all_rel0.5": ("ans", scale(dAns, hAns, 0.5), mask), "ans_all_rel1.0": ("ans", scale(dAns, hAns, 1.0), mask),
                 "rand_ans": ("ans", scale(rAns, hAns, 1.0), mask),
                 "ans_last_nat": ("ans", dAns, mlast), "ans_last_rel0.5": ("ans", scale(dAns, hAns, 0.5), mlast), "ans_last_rel1.0": ("ans", scale(dAns, hAns, 1.0), mlast)}
        for nm, (where, d, mk) in conds.items():
            cp, oth, _, _ = fwd(test, L, where, d, mk, "steer"); dd = d if mk is None else d * mk
            rel = float((dd.norm(dim=-1) / (hIn if where == "in" else hAns).norm(dim=-1).clamp(min=1e-6))[(dd.norm(dim=-1) > 0)].mean())
            rec(srows, L, nm, cp, oth, tgt, ans[test], dict(rel=rel))
        pd.DataFrame(srows).to_csv(f"{OS}/{NAME}_{CON}.csv", index=False)
    print("L", L, {r["cond"]: round(r["hit"], 2) for r in (rows[-3:] if len(pq) else []) + (srows[-11:] if NEWCON else []) if r["cond"] in ("in_patch", "ans_all_patch", "in_nat", "ans_all_nat")}, flush=True)
print("done", flush=True)
