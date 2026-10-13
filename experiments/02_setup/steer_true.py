"""真正的 steering：不借別題的激活，直接加「目標 centroid − 當前 centroid」。python steer_true.py <repo 或路徑> <concept> <名稱>（MK 環境變數指定家族）
- centroid 只用「其他兩個模板、且模型答對」的題計算（被 steer 的模板不參與）；每個位置、每層各自一組 centroid
- 只 steer 模型答對的題；目標 = (答案 + j) mod n，j 在題間輪流取 1…n−1
- 方式：加差值 α=1 / α=2；只加差值的二維平面成分；把 centroid 子空間成分換成目標 centroid；隨機方向（同範數）對照
- 位置：只改最後位置 / 問句後全部位置；每層記錄「輸出 = 目標」比例與目標機率"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, os, numpy as np, pandas as pd, torch
from huggingface_hub import snapshot_download
REPO, CON, NAME = sys.argv[1:4]
path = REPO if os.path.isdir(REPO) else snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"])
sys.argv = ["group.py", os.environ.get("MK", "qwen"), CON, "ans", path]
src_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_lib", "group.py")).read().split("# ---------- hook 與執行 ----------")[0]
src_ = src_.replace('O = os.environ.get("OUT") or f"{DATA}/group/{MNAME}/{CON}/{POS}"', 'O = f"{DATA}/steer_true"')
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
import json; json.dump(dict(acc=float(ok[keep].mean()), acc_wrap=float(ok[keep][wrap[keep] == 1].mean()), n=int(len(keep))), open(f"{O}/{NAME}_{CON}_acc.json", "w"))
rng = np.random.default_rng(0); rows = []
for t in range(3):
    test = [i for i in keep[tt[keep] == t] if ok[i]]; train = [i for i in keep if tt[i] != t and ok[i]]
    if not test: continue
    have = set(ans[train].tolist()); jj = np.array([1 + (r % (n - 1)) for r in range(len(test))]); a_ = ans[test]; b_ = (a_ + jj) % n
    sel = np.array([(x in have) and (y in have) for x, y in zip(a_, b_)]); test = [i for i, s_ in zip(test, sel) if s_]; a_, b_, jj = a_[sel], b_[sel], jj[sel]
    wt = (a_ + jj) >= n  # 目標需要繞回
    if not test: continue
    Pq = Pt[t]; allpos = list(range(P - Pq, P)); POS = {"只改最後位置": [P - 1], "問句後全部位置": allpos}
    for L in range(NL):
        Htr = HT[train, L]; ytr = ans[train]
        C = torch.stack([Htr[ytr == k].mean(0) if (ytr == k).any() else torch.zeros(P, D) for k in range(n)])  # n × P × D（缺的類別不會被用到）
        hv = sorted(have); m = C[hv].mean(0); dif = C[b_] - C[a_]  # T × P × D
        U2 = [torch.linalg.svd(C[hv, j] - m[j], full_matrices=False).Vh[:2].T for j in range(P)]
        Uc = [torch.linalg.svd(C[hv, j] - m[j], full_matrices=False).Vh[: len(hv) - 1].T for j in range(P)]
        h = HT[test, L]
        for pname, js in POS.items():
            mask = torch.zeros(P); mask[js] = 1; mask = mask[None, :, None]
            V = {"加差值 α=1": dif, "加差值 α=2": 2 * dif,
                 "只加二維平面成分": torch.stack([(dif[:, j] @ U2[j]) @ U2[j].T for j in range(P)], 1),
                 "子空間成分換成目標 centroid": torch.stack([((C[b_, j] - h[:, j]) @ Uc[j]) @ Uc[j].T for j in range(P)], 1)}
            r_ = torch.randn_like(dif); V["隨機方向（同範數）"] = r_ / r_.norm(dim=-1, keepdim=True) * dif.norm(dim=-1, keepdim=True)
            for vname, dl in V.items():
                c, o = fwd(test, L, dl * mask)
                hit = torch.cat([c, o[:, None]], 1).argmax(1).numpy() == b_
                rows.append(dict(template=t, layer=L, positions=pname, method=vname, n=len(test), n_wrap=int(wt.sum()), n_classes=len(hv),
                                 hit=float(np.mean(hit)), hit_wrap=float(hit[wt].mean()) if wt.any() else np.nan, hit_nowrap=float(hit[~wt].mean()) if (~wt).any() else np.nan,
                                 p_tgt=float(c[torch.arange(len(test)), torch.tensor(b_)].mean())))
        print("t", t, "L", L, flush=True)
    pd.DataFrame(rows).to_csv(f"{O}/{NAME}_{CON}.csv", index=False)
print("done", flush=True)
