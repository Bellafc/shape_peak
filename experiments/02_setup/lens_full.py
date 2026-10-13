"""全詞表 logit lens（重寫版；原 results/lens_full 的腳本已遺失，按論文附錄 H 的描述重寫）。
最後位置每層殘差 → final norm → lm_head，在全詞表上：
  lens_p = 正確答案（合併寫法，被多個類別共用的 token 不算）的機率，同 lens.py；
  a_lin  = 全詞表 argmax 是正確答案某個寫法第一個 token 的題比例（Billa et al. 的 A_lin）。
python lens_full.py <MK> <模型路徑> <名稱> <concept,concept,...>；輸出 data/lens_full/<名稱>_<concept>.csv"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, os, numpy as np, pandas as pd, torch
MK, MPATH, NAME, CONS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
os.makedirs(f"{DATA}/lens_full", exist_ok=True)
for CON in CONS:
    sys.argv = ["group.py", MK, CON, "ans", MPATH]
    src_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_lib", "group.py")).read().split("# ---------- hook 與執行 ----------")[0]
    src_ = src_.replace('O = os.environ.get("OUT") or f"{DATA}/group/{MNAME}/{CON}/{POS}"; os.makedirs(O, exist_ok=True)', 'O = f"{DATA}/lens_full"')
    if CON != CONS[0]: src_ = src_.replace("model = AutoModelForCausalLM", "_unused = None and AutoModelForCausalLM")  # 同一個模型只載一次
    exec(src_)
    CAP = {}; hooks = [lm.layers[L].register_forward_hook(lambda mod, inp, out, L=L: CAP.__setitem__(L, (out[0] if isinstance(out, tuple) else out)[:, -1])) for L in range(NL)]
    cls_of = torch.full((model.config.get_text_config().vocab_size + 1024,), -1, dtype=torch.long, device="cuda"); cls_of[VI] = VC  # token → 類別（共用 token 已排除）
    P = np.zeros((N, NL)); A = np.zeros((N, NL), bool); by = {}
    for i in range(N): by.setdefault((tt[i], len(R.ids[i])), []).append(i)
    with torch.no_grad():
        for _, g in by.items():
            for b in range(0, len(g), 32):
                sub = g[b:b + 32]; model(input_ids=torch.tensor([R.ids[v] for v in sub], device="cuda"), logits_to_keep=1)
                a_ = torch.tensor(ans[sub], device="cuda")
                for L in range(NL):
                    p = model.lm_head(lm.norm(CAP[L])).float().softmax(-1)
                    c = torch.zeros(len(sub), n, device="cuda").index_add_(1, VC, p[:, VI])
                    P[sub, L] = c[torch.arange(len(sub)), a_].cpu().numpy(); A[sub, L] = (cls_of[p.argmax(-1)] == a_).cpu().numpy()
    for h_ in hooks: h_.remove()
    pd.DataFrame(dict(layer=range(NL), lens_p=P.mean(0), a_lin=A.mean(0))).to_csv(f"{DATA}/lens_full/{NAME}_{CON}.csv", index=False)
    print(NAME, CON, "done", flush=True)
