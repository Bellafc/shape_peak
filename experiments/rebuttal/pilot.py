"""Rebuttal pilots beyond single-word lookups (forward passes and interventions on one GPU).
Usage: python pilot.py <family> <model_path> <name> <pilot>      pilot = arith | kin
  arith  multi-step arithmetic word problems, answer 0-9 (10 classes); conditions: 1/2/3 operations x short/long
         (long = 24 unrelated sentences interleaved, ~300 tokens)
  kin    multiple-choice kinship reasoning (CLUTRR-style), 2 or 3 hops; answer = option letter A-D (4 classes),
         options shuffled so the letter is independent of the relation word; shape is also computed by relation word
Per condition and layer (same protocol as the paper, Appendix D):
  shape            adjacent-layer Procrustes disparity of answer-class centroids at the last position (Eq. 1)
  shape_content    (kin) the same with prompts grouped by the correct relation word
  probe            cross-validated linear identity probe for the answer class (last position)
  patch_in         patching the whole user message from a partner prompt with another answer (success = partner's answer)
  patch_ans        patching every answer position (tokens after the user message)
  patch_last       patching the last position only
  steer            alpha = 1 centroid-difference steering at the answer positions, centroids from the other two templates
Partners have the same template and token length: arith = same story with another start number; kin = same question with
the options in another order. Only correctly answered prompts enter interventions. Output: DATA/rebuttal/pilot/<name>_<pilot>_<cond>.csv"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import os, sys, json, time, random, zlib, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer
FAM, MP, NAME, PILOT = sys.argv[1:5]
SMALL = bool(os.environ.get("SMALL"))  # local smoke test: few prompts, few layers
OUT = f"{DATA}/rebuttal/pilot"; os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- tasks ----------------
FEM = ["Lisa", "Anna", "Emma", "Mary", "Kate", "Sara", "Jane", "Lucy", "Laura", "Helen", "Alice", "Rose"]
MAS = ["Tom", "Sam", "Mark", "Jack", "Paul", "John", "Mike", "Ben", "David", "Peter", "James", "Eric"]
NAMES = FEM + MAS
FILLER = ["The weather was mild that day.", "A bus passed by the corner.", "Somewhere a dog was barking.", "The streetlights had just come on.",
          "A radio was playing quietly in the kitchen.", "The neighbours were painting their fence.", "It had rained a little the night before.",
          "The town library was closed for repairs.", "A group of children walked to school.", "The sky was full of small clouds.",
          "Someone had left a bicycle by the gate.", "The bakery smelled of fresh bread.", "A cat slept on the warm windowsill.",
          "The train to the city was running late.", "Leaves were falling in the park.", "A man was reading a newspaper on a bench.",
          "The market was busier than usual.", "Two birds sat on the telephone wire.", "The river looked calm and grey.",
          "A delivery van stopped across the street.", "The clock on the tower struck ten.", "Music drifted from an open window.",
          "The garden needed watering again.", "A small boat crossed the lake.", "The shop windows were decorated with lights."]

def arith_items(rng, steps, long):
    """Families share template, operations and filler; members differ only in the start number (same token length)."""
    ST = [("{n} has {s} apples.", "{n} buys {k} more.", "{n} gives {k} to a friend.", "How many apples does {n} have now?"),
          ("{n} has {s} marbles.", "{n} finds {k} more.", "{n} loses {k}.", "How many marbles does {n} have now?"),
          ("There are {s} books on {n}'s shelf.", "{n} puts {k} more on the shelf.", "{n} takes {k} off the shelf.",
           "How many books are on {n}'s shelf now?")]
    quota = 4 if SMALL else 16; cnt = np.zeros(10, int); items = []; fam = 0
    while cnt.min() < quota and fam < 5000:
        t = int(rng.integers(3)); n = NAMES[int(rng.integers(len(NAMES)))]
        ops = [(int(rng.choice([1, -1])), int(rng.integers(1, 5))) for _ in range(steps)]
        fill = list(rng.choice(FILLER, 24, replace=False)) if long else []
        valid = []
        for s in range(2, 10):
            v, ok = s, True
            for sg, k in ops:
                v += sg * k; ok &= 0 <= v <= 9
            if ok: valid.append((s, v))
        rng.shuffle(valid); members = [(s, v) for s, v in valid if cnt[v] < quota][:4]
        if len(members) < 2: fam += 1; continue
        for s, v in members:
            sent = [ST[t][0].format(n=n, s=s)] + [ST[t][1 if sg > 0 else 2].format(n=n, k=k) for sg, k in ops]
            if long:  # spread the filler between the sentences, same positions for every member of the family
                per = np.array_split(fill, len(sent)); sent = [x for a, b in zip(sent, per) for x in [a] + list(b)]
            items.append(dict(t=t, fam=fam, ans=v, q=" ".join(sent) + " " + ST[t][3].format(n=n))); cnt[v] += 1
        fam += 1
    return items, [str(d) for d in range(10)], "Answer with only the number. Do not explain."

# kinship: relation of X to Z as (generations up, generations down) from Z
TERM = {(1, 0): ("mother", "father"), (0, 1): ("daughter", "son"), (1, 1): ("sister", "brother"), (2, 0): ("grandmother", "grandfather"),
        (0, 2): ("granddaughter", "grandson"), (2, 1): ("aunt", "uncle"), (1, 2): ("niece", "nephew"), (2, 2): ("cousin", "cousin")}
VOCAB = sorted({w for v in TERM.values() for w in v})
def step(rel, kind, prev):
    """Relation of the next person, who is the <kind> of the person at rel (reached by the move prev). None = ambiguous."""
    u, d = rel
    if kind == "parent":
        if prev == "child": return None                         # the parent is the person we came from (or a second parent)
        return (u, d - 1) if d > 0 else (u + 1, 0)
    if kind == "child": return (u, d + 1) if d > 0 else None   # child of an ancestor is ambiguous (could lead back to Z)
    return (u + 1, 1) if d == 0 and u > 0 else None             # sibling of an ancestor; other siblings are ambiguous

def kin_items(rng, hops):
    FACT = ["{x} is {y}'s {r}.", "{y}'s {r} is {x}.", "{x} is the {r} of {y}."]
    QUES = ["How is {x} related to {z}?", "What is {x} to {z}?", "{x} is {z}'s what?"]
    nb = 6 if SMALL else 50; items = []; b = 0; tries = 0
    while b < nb and tries < 100000:
        tries += 1
        kinds = [str(rng.choice(["parent", "child", "sibling"])) for _ in range(hops)]  # kinds[i]: P_i is P_{i+1}'s <kind>
        rel = {"parent": (1, 0), "child": (0, 1), "sibling": (1, 1)}[kinds[-1]]
        prev = kinds[-1]
        for kd in reversed(kinds[:-1]):
            rel = step(rel, kd, prev) if rel is not None else None; prev = kd
            if rel is None or rel == (0, 0): rel = None; break
        if rel not in TERM: continue
        gens = [int(rng.integers(2)) for _ in range(hops + 1)]  # gender of P_0 .. P_hops (0 = female); names follow gender
        pf, pm = list(rng.permutation(FEM)), list(rng.permutation(MAS)); ppl = [pf.pop() if g == 0 else pm.pop() for g in gens]
        word = lambda kd, g: {"parent": ("mother", "father"), "child": ("daughter", "son"), "sibling": ("sister", "brother")}[kd][g]
        t = b % 3
        facts = " ".join(FACT[t].format(x=ppl[i], y=ppl[i + 1], r=word(kinds[i], gens[i])) for i in range(hops))
        ans_w = TERM[rel][gens[0]]
        opts = [ans_w] + list(rng.choice([w for w in VOCAB if w != ans_w], 3, replace=False))
        letters = list(rng.permutation(4))[:3]  # three orders with different correct letters
        for L in letters:
            o = [w for w in opts[1:]]; rng.shuffle(o); o.insert(int(L), ans_w)
            q = facts + " " + QUES[t].format(x=ppl[0], z=ppl[-1]) + "\n" + "\n".join(f"{'ABCD'[i]}. {w}" for i, w in enumerate(o))
            items.append(dict(t=t, fam=b, ans=int(L), content=ans_w, q=q))
        b += 1
    return items, list("ABCD"), "Answer with only the letter of the correct option. Do not explain."

# ---------------- model ----------------
tok = AutoTokenizer.from_pretrained(MP)
kw = dict(attn_implementation="eager") if FAM == "gemma" else {}
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16 if DEV == "cuda" else torch.float32, device_map=DEV, **kw).eval()
lm = getattr(model.model, "language_model", model.model); NL = len(lm.layers); D = model.config.get_text_config().hidden_size
LAYERS = list(range(NL)) if not SMALL else sorted({0, 1, NL // 2, NL - 2, NL - 1})
BS = 16 if DEV == "cuda" else 4
def chat(sysm, q):
    try: return tok.apply_chat_template([{"role": "system", "content": sysm}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    except Exception: return tok.apply_chat_template([{"role": "user", "content": sysm + "\n\n" + q}], tokenize=False, add_generation_prompt=True)

st = {}
def hook(L):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if st.get("cap"): st["H"][L] = h[:, st["sfx"]].float().cpu()
        if st.get("L") == L and st.get("fn") is not None: st["fn"](h)
        return out
    return f
for L in range(NL): lm.layers[L].register_forward_hook(hook(L))

def prep(items, labels, sysm):
    cand = {}
    for i, w in enumerate(labels):
        for pre in ("", " "):
            ids = tok(pre + w, add_special_tokens=False)["input_ids"]
            if ids: cand.setdefault(ids[0], set()).add(i)
    VAR = {i: [t_ for t_, s_ in cand.items() if s_ == {i}] for i in range(len(labels))}
    assert all(VAR.values()), VAR
    VI = torch.tensor([t_ for i in VAR for t_ in VAR[i]], device=DEV); VC = torch.tensor([i for i in VAR for _ in VAR[i]], device=DEV)
    for it in items:
        p = chat(sysm, it["q"]); s = p.rindex(it["q"]); e = s + len(it["q"])
        enc = tok(p, add_special_tokens=False, return_offsets_mapping=True); off = enc["offset_mapping"]
        it["ids"] = enc["input_ids"]; it["user"] = [j for j, (a, b) in enumerate(off) if a < e and b > s]
        it["sfx"] = [j for j, (a, b) in enumerate(off) if a >= e]  # answer positions: template tokens after the user message
    m = min(len(it["sfx"]) for it in items)
    for it in items: it["sfx"] = it["sfx"][-m:]
    return VI, VC, m

def run(rows, ids, L=None, fn=None, capture=False, n_cls=10, VI=None, VC=None, sfx=None):
    """rows: list of equal-length token lists. Returns class probabilities, 'other' probability, captured states (B x NL x m x D)."""
    x = torch.tensor(ids, device=DEV)
    st.update(L=L, fn=fn, cap=capture, H={}, sfx=torch.tensor(sfx, device=DEV))
    with torch.no_grad(): p = model(input_ids=x, logits_to_keep=1).logits[:, -1].float().softmax(-1)
    c = torch.zeros(len(ids), n_cls, device=DEV).index_add_(1, VC, p[:, VI])
    Hc = torch.stack([st["H"][l] for l in range(NL)], 1).half() if capture else None
    st.update(L=None, fn=None, cap=False)
    return c.cpu(), (1 - c.sum(1)).cpu(), Hc

def batches(idx, items):
    by = {}
    for i in idx: by.setdefault(len(items[i]["ids"]), []).append(i)
    for _, g in sorted(by.items()):
        for b in range(0, len(g), BS): yield g[b:b + BS]

def disp(A, B):
    A = A - A.mean(0); B = B - B.mean(0); A = A / A.norm(); B = B / B.norm()
    Q, _ = torch.linalg.qr(torch.cat([A, B]).T); return float(1 - torch.linalg.svdvals((A @ Q).T @ (B @ Q)).sum() ** 2)

def condition(cond, items, labels, sysm, rng):
    K = len(labels); VI, VC, m = prep(items, labels, sysm); N = len(items)
    y = np.array([it["ans"] for it in items]); tt = np.array([it["t"] for it in items]); fam = np.array([it["fam"] for it in items])
    R = dict(n_cls=K, VI=VI, VC=VC)
    # baseline + activations at the answer positions
    H = torch.zeros(N, NL, m, D, dtype=torch.float16); CP = torch.zeros(N, K); PO = torch.zeros(N)
    for b in batches(range(N), items):
        sf = items[b[0]]["sfx"]; c, po, Hc = run(b, [items[i]["ids"] for i in b], capture=True, sfx=sf, **R)
        CP[b], PO[b], H[b] = c, po, Hc
    pred = torch.cat([CP, PO[:, None]], 1).argmax(1).numpy(); ok = pred == y; acc = float(ok.mean())
    log(cond, f"N={N} tokens~{np.mean([len(it['ids']) for it in items]):.0f} acc={acc:.3f} answer positions={m}")
    Hl = H[:, :, -1].float()  # last position
    # partners: same family (same length and template), different answer, both correct
    pairs = []
    for f_ in np.unique(fam):
        g = [i for i in np.where(fam == f_)[0] if ok[i]]
        for a in g:
            bs = [b for b in g if y[b] != y[a]]
            if bs: pairs.append((a, int(rng.choice(bs))))
    rng.shuffle(pairs); pairs = pairs[:(12 if SMALL else 120)]
    # steering set: correct prompts, random other target; centroids from the other templates (correct prompts only)
    have = lambda i, c: bool(((tt != tt[i]) & ok & (y == c)).any())
    SS = [i for i in np.where(ok)[0] if have(i, y[i])]; rng.shuffle(SS); tgt = {}
    for i in SS:
        cs = [c for c in range(K) if c != y[i] and have(i, c)]
        if cs: tgt[i] = int(rng.choice(cs))
    SS = [i for i in SS if i in tgt][:(12 if SMALL else 120)]
    rows = []; prev = prevc = None
    for L in LAYERS:
        C = torch.stack([Hl[y == c, L].mean(0) for c in range(K)])
        r = dict(cond=cond, layer=L, N=N, acc=acc, n_pairs=len(pairs), n_steer=len(SS), shape=disp(prev, C) if prev is not None else np.nan)
        if PILOT == "kin":
            cw = np.array([it["content"] for it in items]); ws = sorted(set(cw)); Cc = torch.stack([Hl[cw == w, L].mean(0) for w in ws])
            r["shape_content"] = disp(prevc, Cc) if prevc is not None else np.nan; prevc = Cc
        prev = C
        X = Hl[:, L].numpy(); X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = PCA(min(64, N - 1), random_state=0).fit_transform(X)
        nf = int(min(5, np.bincount(y, minlength=K)[np.bincount(y, minlength=K) > 0].min()))
        r["probe"] = float(cross_val_score(LogisticRegression(max_iter=2000, C=0.1), X, y, cv=StratifiedKFold(max(nf, 2), shuffle=True, random_state=0)).mean()) if nf >= 2 else np.nan
        # patching: A and B rows in one batch; at layer L copy B's states into A's rows at the chosen positions
        for nm in ("patch_in", "patch_ans", "patch_last"):
            hit = []
            by = {}
            for a, b in pairs: by.setdefault(len(items[a]["ids"]), []).append((a, b))
            for _, g in by.items():
                for k in range(0, len(g), BS // 2):
                    sub = g[k:k + BS // 2]; A_ = [a for a, _ in sub]; B_ = [b for _, b in sub]; n = len(sub)
                    pos = {"patch_in": items[A_[0]]["user"], "patch_ans": items[A_[0]]["sfx"], "patch_last": [len(items[A_[0]]["ids"]) - 1]}[nm]
                    pos_t = torch.tensor(pos, device=DEV); ia = torch.arange(n, device=DEV); ib = ia + n
                    def fn(h, pos_t=pos_t, ia=ia, ib=ib): h[ia[:, None], pos_t] = h[ib[:, None], pos_t].clone()
                    c, po, _ = run(None, [items[i]["ids"] for i in A_ + B_], L=L, fn=fn, sfx=items[A_[0]]["sfx"], **R)
                    pr = torch.cat([c, po[:, None]], 1).argmax(1).numpy()[:n]; hit += list(pr == y[B_])
            r[nm] = float(np.mean(hit)) if hit else np.nan
        # steering alpha = 1 at every answer position, per-position centroids from the other templates
        hit = []
        for b in batches(SS, items):
            dl = []
            for i in b:
                msk = (tt != tt[i]) & ok
                ca = H[msk & (y == tgt[i]), L].float().mean(0); co = H[msk & (y == y[i]), L].float().mean(0)
                dl.append(ca - co)
            dl = torch.stack(dl).to(DEV); sf = torch.tensor(items[b[0]]["sfx"], device=DEV)
            def fn(h, dl=dl, sf=sf): h[:, sf] += dl.to(h.dtype)
            c, po, _ = run(None, [items[i]["ids"] for i in b], L=L, fn=fn, sfx=items[b[0]]["sfx"], **R)
            pr = torch.cat([c, po[:, None]], 1).argmax(1).numpy(); hit += list(pr == np.array([tgt[i] for i in b]))
        r["steer"] = float(np.mean(hit)) if hit else np.nan
        rows.append(r)
        if L % 4 == 0 or SMALL: log(cond, f"L{L}", {k: round(v, 2) for k, v in r.items() if k in ("shape", "probe", "patch_in", "patch_ans", "steer")})
    pd.DataFrame(rows).to_csv(f"{OUT}/{NAME}_{PILOT}_{cond}.csv", index=False)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    if PILOT == "arith":
        conds = [(f"ops{s}_{'long' if lg else 'short'}", lambda r, s=s, lg=lg: arith_items(r, s, lg)) for lg in (False, True) for s in (1, 2, 3)]
    else:
        conds = [(f"hop{h}", lambda r, h=h: kin_items(r, h)) for h in (2, 3)]
    if SMALL: conds = conds[:1] + conds[-1:]
    for cond, mk in conds:
        if os.path.exists(f"{OUT}/{NAME}_{PILOT}_{cond}.csv"): log("skip", cond); continue
        items, labels, sysm = mk(np.random.default_rng(zlib.crc32(cond.encode())))
        condition(cond, items, labels, sysm, np.random.default_rng(1))
    log("done", PILOT)
