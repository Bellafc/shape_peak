"""通宵跑次的單組核心流程：python group.py <model> <concept> <position> <model_path>
model ∈ llama / qwen / gemma；concept ∈ weekday / month / hour / letter；position ∈ ans / concept
每層：形狀指標、身份探針、子空間 / 整個殘差 / 隨機 2 維 patching、消融（繞回 / 非繞回）+ 消融後身份探針、旋轉 steering（對照隨機平面）。
每做完一層就寫一次 per_layer.csv；最後寫 summary.md 與 row.json（給總表用）。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import os, sys, json, time, string, numpy as np, pandas as pd, torch
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer
MKEY, CON, POS, MPATH = sys.argv[1:5]
NEWCONS = ("city", "emotion", "color", "continent", "capital2hop", "ctxcolor", "month2step")
MNAME = os.environ.get("MNAME_OVERRIDE") or {"llama": "Llama-3.1-8B-Instruct", "qwen": "Qwen2.5-7B-Instruct", "gemma": "Gemma-2-9B-it", "mistral": "Mistral-7B-Instruct-v0.3"}[MKEY]
CNAME = {"weekday": "星期", "month": "月份", "hour": "小時", "letter": "字母", "city": "城市", "emotion": "情緒", "color": "顏色", "continent": "國家→洲", "capital2hop": "首都→國家→洲", "ctxcolor": "上下文顏色", "month2step": "兩步月份"}[CON]; PNAME = {"ans": "答案位置", "concept": "概念 token 位置"}[POS]
O = os.environ.get("OUT") or f"{DATA}/group/{MNAME}/{CON}/{POS}"; os.makedirs(O, exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
# ---------- 概念定義 ----------
KW = ["one", "two", "three", "four", "five", "six"]
if CON == "weekday":
    ITEMS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]; UNIT = "day"
    T = ["What day is {k} after {x}?", "Today is {x}. What day will it be in {k}?", "Starting from {x} and counting {k} forward, what day is it?"]
    SYS = "Answer with only the name of the day. Do not explain."
    FORMS = lambda i: [ITEMS[i], ITEMS[i].lower()]
elif CON == "month":
    ITEMS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]; UNIT = "month"
    T = ["What month is {k} after {x}?", "It is {x} now. What month will it be in {k}?", "Starting from {x} and counting {k} forward, what month is it?"]
    SYS = "Answer with only the name of the month. Do not explain."
    FORMS = lambda i: [ITEMS[i], ITEMS[i].lower()]
elif CON == "hour":
    ITEMS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]; UNIT = "hour"
    T = ["What time is {k} after {x} o'clock?", "It is {x} o'clock now. What time will it be in {k}?", "Starting at {x} o'clock and counting {k} forward, what time is it?"]
    SYS = "Answer with only the hour. Do not explain."
    FORMS = lambda i: [ITEMS[i], ITEMS[i].capitalize(), str(i + 1)]
elif CON in NEWCONS:
    import importlib.util; sp = importlib.util.spec_from_file_location("nc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_lib", "new_concepts.py")); nc = importlib.util.module_from_spec(sp); sp.loader.exec_module(nc)
    CUES, T, SYS, EXTRA = nc.NEWSET[CON]
    ITEMS = list(CUES); UNIT = ""
    FORMS = lambda i: [ITEMS[i], ITEMS[i].capitalize()] + [w for e in EXTRA.get(ITEMS[i], []) for w in (e, e.capitalize())]
else:
    ITEMS = list(string.ascii_uppercase); UNIT = "letter"
    T = ["What letter comes {k} after {x}?", "Starting from the letter {x} and counting {k} forward in the alphabet, what letter is it?", "In the alphabet, which letter is {k} after {x}?"]
    SYS = "Answer with only the letter. Do not explain."
    FORMS = lambda i: [ITEMS[i], ITEMS[i].lower()]
n = len(ITEMS); kd = lambda k: f"{KW[k - 1]} {UNIT}" + ("" if k == 1 else "s")
# ---------- 模型 ----------
tok = AutoTokenizer.from_pretrained(MPATH)
kw = dict(attn_implementation="eager") if MKEY == "gemma" else {}
model = AutoModelForCausalLM.from_pretrained(MPATH, dtype=torch.bfloat16, device_map="cuda", **kw).eval()
lm = getattr(model.model, "language_model", model.model)  # Gemma 3 是多模態封裝，文字解碼層在 language_model 裡
NL = len(lm.layers); D = model.config.get_text_config().hidden_size
def chat_(q):
    try: return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception: return tok.apply_chat_template([{"role": "user", "content": SYS + "\n\n" + q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)  # Gemma 2 無 system 角色
def chat(q):  # Llama 2：[/INST] 後模型先吐一個單獨的「▁」再吐答案；把空格放進提示，答案位置的下一個 token 才是答案本身
    p = chat_(q); return p + " " if p.endswith("[/INST]") else p
MARK = "⁣QQ⁣"
SUB = bool(os.environ.get("SUB"))  # 小樣本驗證用（預設關閉，與原協議相同）
rows = []
NEWCON = CON in NEWCONS
for t, tp in enumerate(T if NEWCON else []):  # 新概念：src = 線索編號、k = 類內線索序號（配對時保證不同線索），沒有繞回
    for i in range(n):
        for j, cue in enumerate(CUES[ITEMS[i]]):
            if SUB and j >= (len(CUES[ITEMS[i]]) + 1) // 2: continue  # 小樣本：每類只用前一半線索
            off = tp.format(x=MARK).index(MARK); q = tp.format(x=cue); q = q[0].upper() + q[1:]; p = chat(q); s = p.rindex(q) + off
            enc = tok(p, add_special_tokens=False, return_offsets_mapping=True)
            cpos = [jj for jj, (a, b) in enumerate(enc["offset_mapping"]) if a < s + len(cue) and b > s][-1]
            rows.append(dict(t=t, src=i * 100 + j, k=j + 1, ans=i, wrap=0, prompt=p, ids=enc["input_ids"], cpos=cpos))
for t, tp in enumerate([] if NEWCON else T):
    for i in range(n):
        for k in range(1, 7):
            if SUB and k % 2 == 0: continue  # 小樣本：只用 k = 1, 3, 5
            off = tp.format(k=kd(k), x=MARK).index(MARK); q = tp.format(k=kd(k), x=ITEMS[i]); p = chat(q); s = p.rindex(q) + off
            enc = tok(p, add_special_tokens=False, return_offsets_mapping=True)
            cpos = [j for j, (a, b) in enumerate(enc["offset_mapping"]) if a < s + len(ITEMS[i]) and b > s][-1]
            rows.append(dict(t=t, src=i, k=k, ans=(i + k) % n, wrap=int(i + k >= n), prompt=p, ids=enc["input_ids"], cpos=cpos))
R = pd.DataFrame(rows); N = len(R); src, ans, kk, wrap, tt = R.src.values, R.ans.values, R.k.values, R.wrap.values, R.t.values
LAB = ans if POS == "ans" else src
# 答案 token：各寫法（有 / 無前置空格）若為單一 token 就收；多 token 寫法收第一個 token；被兩個以上類別共用的 token 丟掉
cand = {}
for i in range(n):
    for f in FORMS(i):
        for pre in ("", " "):
            ids = tok(pre + f, add_special_tokens=False)["input_ids"]
            if ids: cand.setdefault(ids[0], set()).add(i)
VAR = {i: sorted(t_ for t_, s_ in cand.items() if s_ == {i}) for i in range(n)}
assert all(VAR[i] for i in range(n)), f"有類別沒有可用的答案 token：{[ITEMS[i] for i in range(n) if not VAR[i]]}"
VI = torch.tensor([t_ for i in range(n) for t_ in VAR[i]], device="cuda"); VC = torch.tensor([i for i in range(n) for _ in VAR[i]], device="cuda")
# ---------- hook 與執行 ----------
st = {"layer": None, "cap": False}
def hook(L):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1: return out
        ar = torch.arange(h.shape[0], device=h.device)
        if st.get("cap"): st["H"][L] = h[ar, st["pos"]].float().cpu()
        if st["layer"] == L: h[ar, st["pos"]] = st["new"].to(h.dtype)
        return out
    return f
for L in range(NL): lm.layers[L].register_forward_hook(hook(L))
BS = 32 if MKEY == "gemma" else 64
GROUPS = {}
for i in range(N): GROUPS.setdefault((tt[i], len(R.ids[i])), []).append(i)
def run(idx, L=None, new=None, capture=False):
    """idx 任意；按（模板, 長度）分批，不需要 padding。回傳 n 類機率（N×n）與「其他」機率，capture 時也回傳該位置各層狀態。"""
    idx = list(idx); where = {v: j for j, v in enumerate(idx)}; cp = torch.zeros(len(idx), n); po = torch.zeros(len(idx))
    Hc = torch.zeros(len(idx), NL, D) if capture else None
    by = {}
    for v in idx: by.setdefault((tt[v], len(R.ids[v])), []).append(v)
    for _, g in by.items():
        for b in range(0, len(g), BS):
            sub = g[b:b + BS]; rows_ = [where[v] for v in sub]
            ids = torch.tensor([R.ids[v] for v in sub], device="cuda")
            pos = torch.tensor([len(R.ids[v]) - 1 if POS == "ans" else R.cpos[v] for v in sub], device="cuda")
            st.update(layer=L, new=None if new is None else new[rows_].cuda(), cap=capture, H={}, pos=pos)
            with torch.no_grad(): p = model(input_ids=ids, logits_to_keep=1).logits[:, -1].float().softmax(-1)
            c = torch.zeros(len(sub), n, device="cuda").index_add_(1, VC, p[:, VI]); cp[rows_] = c.cpu(); po[rows_] = (1 - c.sum(1)).cpu()
            if capture:
                for l_ in range(NL): Hc[rows_, l_] = st["H"][l_]
            st.update(layer=None, cap=False)
    return cp, po, Hc
pred_of = lambda cp, po: torch.cat([cp, po[:, None]], 1).argmax(1).numpy()
def LD(cp, tgt, base): l = (cp + 1e-12).log(); r_ = torch.arange(len(cp)); return (l[r_, list(tgt)] - l[r_, list(base)]).numpy()
res_row = dict(model=MNAME, concept=CNAME, position=PNAME, NL=NL, N=N, status="running")
def save_row(**kw_): res_row.update(kw_); json.dump(res_row, open(f"{O}/row.json", "w"), ensure_ascii=False, default=float)
# ---------- 煙霧測試（12 題，貪婪解碼） ----------
rng = np.random.default_rng(0); sm = rng.choice(N, 12, replace=False); gen_ok = []; first_forms = []
def parse_ok(text, a):
    s = text.strip().lower().lstrip("\"'*").replace("o'clock", "").strip()
    if CON == "letter": return len(s) > 0 and s[0] == ITEMS[a].lower() and (len(s) == 1 or not s[1].isalpha())
    if CON == "hour": w = s.split()[0].strip(".,!") if s.split() else ""; return w in (ITEMS[a], str(a + 1))
    if CON in NEWCONS: return any(s.startswith(f.lower()) for f in FORMS(a))
    return s.startswith(ITEMS[a].lower())
for i in sm:
    ids = torch.tensor([R.ids[i]], device="cuda")
    with torch.no_grad(): g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=8, do_sample=False)
    txt = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True); gen_ok.append(parse_ok(txt, ans[i])); first_forms.append(repr(tok.decode(g[0, ids.shape[1]])))
smoke_acc = float(np.mean(gen_ok))
log(f"煙霧測試：生成準確率 {smoke_acc:.2f}；第一個 token 的形式 {sorted(set(first_forms))}")
# ---------- 基準與激活 ----------
CP0, PO0, H = run(range(N), capture=True); ok0 = pred_of(CP0, PO0) == ans
acc, acc_w, acc_nw = ok0.mean(), ok0[wrap == 1].mean(), ok0[wrap == 0].mean()
save_row(smoke_acc=smoke_acc, first_forms=sorted(set(first_forms)), acc=acc, acc_wrap=acc_w, acc_nowrap=acc_nw, other=float(PO0.mean()))
log(f"第一個 token 準確率 {acc:.2f}（繞回 {acc_w:.2f} / 非繞回 {acc_nw:.2f}）；「其他」{PO0.mean():.3f}")
if (smoke_acc < 0.5 or acc < 0.5) and not os.environ.get("FORCE"):
    save_row(status=f"skipped：準確率過低（煙霧測試 {smoke_acc:.2f}，第一個 token {acc:.2f}）"); open(f"{O}/summary.md", "w").write(f"# {MNAME} {CNAME} {PNAME}\n\n跳過：{res_row['status']}\n"); sys.exit(0)
# ---------- 每層 ----------
poly = torch.tensor([[np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)] for i in range(n)], dtype=torch.float32)
def disp(A, B):
    A = A - A.mean(0); B = B - B.mean(0); A = A / A.norm(); B = B / B.norm()
    if A.shape[1] > n: Q, _ = torch.linalg.qr(torch.cat([A, B]).T); A, B = A @ Q, B @ Q
    return float(1 - torch.linalg.svdvals(A.T @ B).sum() ** 2)
cd = lambda a, b: min(abs(a - b) % n, n - abs(a - b) % n)
from sklearn.model_selection import GroupKFold
def id_probe(X, y):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = PCA(64, random_state=0).fit_transform(X)
    if NEWCON:  # 新概念：按線索分組交叉驗證（測試線索在訓練時沒見過），避免探針只記住「線索字面 → 答案」
        return float(cross_val_score(LogisticRegression(max_iter=2000, C=0.1), X, y, cv=GroupKFold(6), groups=src).mean())
    return float(cross_val_score(LogisticRegression(max_iter=2000, C=0.1), X, y, cv=5).mean())
P_ = []
for t in range(3):
    idx = np.where(tt == t)[0]
    while sum(1 for p in P_ if p[2] == t) < (14 if t < 2 else 12):
        a, b = rng.choice(idx, 2, replace=False)
        if POS == "ans" and ans[a] != ans[b] and src[a] != src[b] and kk[a] != kk[b]: P_.append((a, b, t))
        if POS == "concept" and kk[a] == kk[b] and src[a] != src[b]: P_.append((a, b, t))
A_, B_ = np.array([p[0] for p in P_]), np.array([p[1] for p in P_]); tgt, base = ans[B_], ans[A_]
lA, lB = LD(CP0[A_], tgt, base), LD(CP0[B_], tgt, base)
RS = np.sort(np.concatenate([rng.choice(np.where(tt == t)[0], min(36, (tt == t).sum()), replace=False) for t in range(3)]))  # 旋轉用的子集（每模板 ≤ 36 題）
JS = list(range(1, n)) if n <= 12 else [1, 3, 5, 8, 13, 18, 21, 25]
res = []; prevP = prevC = None
LAYERS = [int(x) for x in os.environ["LAYERS"].split(",")] if os.environ.get("LAYERS") else range(NL)  # 只在測試時用
for L in LAYERS:
    C = torch.stack([H[LAB == a, L].mean(0) for a in range(n)]); m = C.mean(0)
    _, S, Vh = torch.linalg.svd(C - m, full_matrices=False); U = Vh[:2].T; P = (C - m) @ U
    adj = np.mean([float((P[i] - P[i + 1]).norm()) for i in range(n - 1)])
    r = dict(layer=L, ev2=float((S[:2] ** 2).sum() / (S ** 2).sum()), closure=float((P[n - 1] - P[0]).norm()) / adj, poly=disp(P, poly),
             change2d=disp(P, prevP) if prevP is not None else np.nan, change_full=disp(C, prevC) if prevC is not None else np.nan)
    prevP, prevC = P, C
    X = H[:, L].numpy()
    for held, nm in ((n - 1, "last"), (0, "first")):
        tr = LAB != held; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        pp = RidgeCV(alphas=np.logspace(0, 5, 11)).fit((X[tr] - mu) / sd, np.stack([np.sin(2 * np.pi * LAB[tr] / n), np.cos(2 * np.pi * LAB[tr] / n)], 1)).predict((X[~tr] - mu) / sd).mean(0)
        r[f"loo_{nm}"] = (np.arctan2(pp[0], pp[1]) % (2 * np.pi)) / (2 * np.pi) * n; r[f"loo_{nm}_err"] = cd(r[f"loo_{nm}"], held)
    r["identity"] = id_probe(X, LAB)
    # 因果：子空間 / 整個殘差 / 隨機 2 維
    Rr = torch.tensor(np.linalg.qr(rng.standard_normal((D, 2)))[0], dtype=torch.float32)
    hA, hB = H[A_, L], H[B_, L]; dU = ((hB - hA) @ U) @ U.T; dR = ((hB - hA) @ Rr) @ Rr.T
    dR = dR / dR.norm(dim=1, keepdim=True).clamp(min=1e-6) * dU.norm(dim=1, keepdim=True)
    for nm, new in (("subspace", hA + dU), ("full", hB), ("random", hA + dR)):
        cp, po, _ = run(A_, L, new)
        r[f"{nm}_norm"] = float(np.mean((LD(cp, tgt, base) - lA) / (lB - lA))); r[f"{nm}_flip"] = float(np.mean(pred_of(cp, po) == tgt))
    # 消融 U 分量 → centroid 平均
    h = H[:, L]; cp, po, _ = run(range(N), L, h - ((h - m) @ U) @ U.T); okA = pred_of(cp, po) == ans
    r.update(abl_acc=okA.mean(), abl_wrap=okA[wrap == 1].mean(), abl_nowrap=okA[wrap == 0].mean(), abl_identity=id_probe((h - ((h - m) @ U) @ U.T).numpy(), LAB))
    # 旋轉 steering（Δ = j·360°/n），對照隨機平面
    fol, rfol = [], []; hS = H[RS, L]; z = (hS - m) @ U; z2 = (hS - m) @ Rr
    for j in JS:
        a_ = 2 * np.pi * j / n; Rm = torch.tensor([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]], dtype=torch.float32)
        zr = z @ Rm.T; near = torch.cdist(zr, P).argmin(1).numpy(); exp_ = near if POS == "ans" else (near + kk[RS]) % n
        dl = (zr - z) @ U.T; cp, po, _ = run(RS, L, hS + dl); fol += list(pred_of(cp, po) == exp_)
        d2 = (z2 @ Rm.T - z2) @ Rr.T; d2 = d2 / d2.norm(dim=1, keepdim=True).clamp(min=1e-6) * dl.norm(dim=1, keepdim=True)
        cp, po, _ = run(RS, L, hS + d2); rfol += list(pred_of(cp, po) == exp_)
    r.update(follow=np.mean(fol), rand_follow=np.mean(rfol))
    res.append(r); pd.DataFrame(res).to_csv(f"{O}/per_layer.csv", index=False); log(f"L{L}", {k_: round(float(v), 2) for k_, v in r.items() if k_ in ("identity", "change_full", "subspace_norm", "follow")})
# ---------- 自動挑層與摘要 ----------
df = pd.DataFrame(res)
id_layer = next((int(x.layer) for x in df.itertuples() if x.identity >= 0.5), None)
cand_ = df[(df.layer >= max(id_layer or 0, 1))].dropna(subset=["change_full"])
shape_peak = int(cand_.loc[cand_.change_full.idxmax(), "layer"]) if len(cand_) else None
onset = next((int(x.layer) for x in df.itertuples() if x.subspace_norm >= 0.3), None)
handoff = None
if onset is not None and POS == "concept": handoff = next((int(x.layer) for x in df.itertuples() if x.layer > onset and x.subspace_norm < 0.1), None)
plat = df[(df.layer >= onset) & ((df.layer < handoff) if handoff is not None else True)] if onset is not None else df.iloc[0:0]
ref = plat if len(plat) else df.tail(3)
wrap_drop, nowrap_drop = acc_w - plat.abl_wrap.mean(), acc_nw - plat.abl_nowrap.mean()
closed = bool(ref.closure.mean() <= 1.5 and ref.loo_last_err.mean() <= 0.75)
save_row(status="ok", id_layer=id_layer, shape_peak=shape_peak, onset=onset, handoff=handoff,
         plat_patch=plat.subspace_norm.mean(), plat_follow=plat.follow.mean(), plat_rand_follow=plat.rand_follow.mean(),
         abl_wrap=plat.abl_wrap.mean(), abl_nowrap=plat.abl_nowrap.mean(), wrap_drop=wrap_drop, nowrap_drop=nowrap_drop,
         closure=ref.closure.mean(), loo_err=ref.loo_last_err.mean(), closed=closed,
         match=(shape_peak is not None and onset is not None and abs(shape_peak - onset) <= 1),
         onset_frac=None if onset is None else onset / NL, wrap_specific=bool(wrap_drop - nowrap_drop >= 0.2) if len(plat) else None, minutes=(time.time() - T0) / 60)
out = [f"# {MNAME}｜{CNAME}｜{PNAME}\n", f"- 模型路徑 `{MPATH}`；chat template + 系統指令「{SYS}」" + ("（Gemma 2 無 system 角色，指令併入 user 訊息）" if MKEY == "gemma" else ""),
       f"- 題數 {N}（{n} 個起點 × k = 1–6 × 3 個模板）；共 {NL} 層；答案 token 變體數 { {ITEMS[i]: len(VAR[i]) for i in range(n)} }",
       f"- 煙霧測試（12 題生成）準確率 {smoke_acc:.2f}；第一個 token 形式 {sorted(set(first_forms))}",
       f"- 第一個 token 準確率 {acc:.2f}（繞回 {acc_w:.2f} / 非繞回 {acc_nw:.2f}），「其他」機率 {PO0.mean():.3f}",
       f"- 旋轉 steering 用 {len(RS)} 題子集，j = {JS}",
       f"\n## 自動挑層\n- 身份探針首次 ≥ 0.5：L{id_layer}\n- 形狀變化峰值（身份探針 ≥ 0.5 之後的全維 Procrustes 最大）：L{shape_peak}\n- patching 起點（子空間首次 ≥ 0.3）：L{onset}" + (f"；交接（效果 < 0.1）：L{handoff}" if handoff is not None else ""),
       f"- 形狀峰值 == patching 起點（±1）：{res_row['match']}；起點佔總深度 {res_row['onset_frac']}",
       f"- 平台期：子空間 patching {plat.subspace_norm.mean():.2f}；旋轉跟著轉 {plat.follow.mean():.2f}（隨機平面 {plat.rand_follow.mean():.2f}）；消融後準確率 繞回 {plat.abl_wrap.mean():.2f} / 非繞回 {plat.abl_nowrap.mean():.2f}；繞回專一：{res_row['wrap_specific']}",
       f"- 閉合：閉合比 {ref.closure.mean():.2f}、留出最後一類探針誤差 {ref.loo_last_err.mean():.2f} → {closed}",
       "\n## 每層\n", df.round(3).to_csv(sep="|", index=False)]
open(f"{O}/summary.md", "w").write("\n".join(out)); log("done")
