"""答案位置各層 centroid：二維平面能解釋多少整體形狀？跳升層的形狀變化是不是藏在 PC3 以後？python dims.py <model> <repo> <concept> <名稱>
每層：各 PC 變異佔比；層間 Procrustes 三種：(a) 全維（原指標，按方差加權）(b) 去掉當層 PC1–2 後的其餘空間 (c) 白化後全維（各維等權）；各傅立葉頻率的變異佔比。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, os, numpy as np, pandas as pd, torch
from huggingface_hub import snapshot_download
MK, REPO, CON, NAME = sys.argv[1:5]
path = snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"])
sys.argv = ["group.py", MK, CON, "ans", path]
src_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_lib", "group.py")).read().split("# ---------- 煙霧測試")[0]
src_ = src_.replace('O = os.environ.get("OUT") or f"{DATA}/group/{MNAME}/{CON}/{POS}"', f'O = f"{DATA}/dims"')
exec(src_)
_, _, H = run(range(N), capture=True)
torch.save(dict(H=H.half() if H.abs().max() < 6e4 else H.bfloat16(), ans=ans, src=src, k=kk, wrap=wrap, t=tt), f"{DATA}/acts/{NAME}_{CON}_acts.pt")
def proc(A, B):
    A = A - A.mean(0); B = B - B.mean(0); A = A / A.norm(); B = B / B.norm()
    Q, _ = torch.linalg.qr(torch.cat([A, B]).T); A, B = A @ Q, B @ Q
    return float(1 - torch.linalg.svdvals(A.T @ B).sum() ** 2)
d_ = np.arange(n); FS = list(range(1, n // 2 + 1))
def fpow(Cc):
    tot = float((Cc ** 2).sum()); out = []
    for f in FS:
        vs = [torch.tensor(np.cos(2 * np.pi * f * d_ / n), dtype=torch.float32)]
        if 2 * f != n: vs.append(torch.tensor(np.sin(2 * np.pi * f * d_ / n), dtype=torch.float32))
        Bm = torch.linalg.qr(torch.stack(vs, 1))[0]  # n × (1 或 2)，在「類別」軸上投影
        out.append(float(((Bm.T @ Cc) ** 2).sum()) / tot)
    return out
rows = []; prev = None
for L in range(NL):
    C = torch.stack([H[ans == a, L].mean(0) for a in range(n)]); Cc = C - C.mean(0)
    _, S, Vh = torch.linalg.svd(Cc, full_matrices=False); ev = (S ** 2 / (S ** 2).sum()).numpy()
    rest = Cc - (Cc @ Vh[:2].T) @ Vh[:2]                     # 去掉當層 PC1–2
    Zw = (Cc @ Vh[: n - 1].T) / S[: n - 1]                      # 白化座標（n−1 維，各維等權）
    r = dict(layer=L, ev1=ev[0], ev2=ev[:2].sum(), ev3_4=ev[2:4].sum(), ev5p=ev[4:].sum(), rest_frac=float((rest ** 2).sum() / (Cc ** 2).sum()))
    for f, p in zip(FS, fpow(Cc)): r[f"f{f}"] = p
    if prev is not None:
        r["chg_full"] = proc(Cc, prev["Cc"]); r["chg_rest"] = proc(rest, prev["rest"]); r["chg_white"] = proc(Zw, prev["Zw"])
        r["chg_fspec"] = float(np.abs(np.array(fpow(Cc)) - np.array(fpow(prev["Cc"]))).sum())
    prev = dict(Cc=Cc, rest=rest, Zw=Zw); rows.append(r)
pd.DataFrame(rows).to_csv(f"{DATA}/dims/{NAME}_{CON}.csv", index=False); print("done", flush=True)
