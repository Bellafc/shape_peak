"""logit lens（不需干預）：最後位置每層殘差 → final norm → lm_head，正確答案（合併寫法）的機率與 top-1 正確率。
python lens.py <MK> <模型路徑> <名稱> <concept,concept,...>
用途：(1) 判斷答案從哪一層開始可讀（取代身份探針，避免線索字面造成的過早觸發）；(2) 當作另一個不需校準的選層規則。
輸出 results/lens/<名稱>_<concept>.csv（layer, lens_p, lens_top）"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, os, numpy as np, pandas as pd, torch
MK, MPATH, NAME, CONS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
os.makedirs(f"{DATA}/lens", exist_ok=True)
model = None
for CON in CONS:
    sys.argv = ["group.py", MK, CON, "ans", MPATH]
    src_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_lib", "group.py")).read().split("# ---------- hook 與執行 ----------")[0]
    src_ = src_.replace('O = os.environ.get("OUT") or f"{DATA}/group/{MNAME}/{CON}/{POS}"; os.makedirs(O, exist_ok=True)', 'O = f"{DATA}/lens"')
    if model is not None:  # 同一模型只載入一次
        src_ = src_.replace("model = AutoModelForCausalLM.from_pretrained(MPATH, dtype=torch.bfloat16, device_map=\"cuda\", **kw).eval()", "pass")
    exec(src_)
    CAP = {}
    hooks = [lm.layers[L].register_forward_hook(lambda mod, inp, out, L=L: CAP.__setitem__(L, (out[0] if isinstance(out, tuple) else out)[:, -1])) for L in range(NL)]
    P = np.zeros((N, NL)); TOP = np.zeros((N, NL), bool)
    by = {}
    for i in range(N): by.setdefault(len(R.ids[i]), []).append(i)
    with torch.no_grad():
        for _, g in by.items():
            for b in range(0, len(g), 32):
                sub = g[b:b + 32]; model(input_ids=torch.tensor([R.ids[i] for i in sub], device="cuda"), logits_to_keep=1)
                for L in range(NL):
                    p = model.lm_head(lm.norm(CAP[L])).float().softmax(-1); c = torch.zeros(len(sub), n, device="cuda").index_add_(1, VC, p[:, VI])
                    oth = 1 - c.sum(1); P[sub, L] = c[torch.arange(len(sub)), torch.tensor(ans[sub], device="cuda")].cpu().numpy()
                    TOP[sub, L] = (torch.cat([c, oth[:, None]], 1).argmax(1).cpu().numpy() == ans[sub])
    for h in hooks: h.remove()
    pd.DataFrame(dict(layer=range(NL), lens_p=P.mean(0), lens_top=TOP.mean(0))).to_csv(f"{DATA}/lens/{NAME}_{CON}.csv", index=False)
    print(NAME, CON, "done", flush=True)
