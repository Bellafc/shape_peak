"""Rebuttal: store answer-position activations (every layer) for one task, on the half-size prompt set (SUB=1).
Usage: SUB=1 python extract_acts.py <family> <task> <model_path> <name>
Reuses the shared prefix of 00_lib/group.py (prompts, chat template, hooks, batched forward), exactly as 02_setup/dims.py does.
Forward passes only: no patching or steering. Output: DATA/rebuttal/acts/<name>_<task>.pt"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, os, torch
MK, CON, MP, NAME = sys.argv[1:5]
OUTD = f"{DATA}/rebuttal/acts"; os.makedirs(OUTD, exist_ok=True)
sys.argv = ["group.py", MK, CON, "ans", MP]
src_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_lib", "group.py")).read().split("# ---------- 煙霧測試")[0]
src_ = src_.replace('O = os.environ.get("OUT") or f"{DATA}/group/{MNAME}/{CON}/{POS}"', f'O = "{OUTD}"')
exec(src_)
CP0, PO0, H = run(range(N), capture=True)
acc = float((pred_of(CP0, PO0) == ans).mean())
torch.save(dict(H=H.half() if H.abs().max() < 6e4 else H.bfloat16(), ans=ans, src=src, k=kk, wrap=wrap, t=tt, acc=acc, sub=SUB),
           f"{OUTD}/{NAME}_{CON}.pt")
print(f"done {CON}: N={N} layers={NL} classes={n} first-token acc={acc:.3f} sub={SUB}", flush=True)
