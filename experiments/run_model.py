"""Run every GPU experiment for one or more models, spread over several GPUs (one job per GPU at a time).
Usage:  python run_model.py "<GPU ids>" <family>:<name>:<HF repo> [...]
        e.g. python run_model.py "0 1 2 3" qwen:Qwen2.5-14B-Instruct:Qwen/Qwen2.5-14B-Instruct phi:Phi-4-14B:microsoft/phi-4
<family> selects model-specific handling in 00_lib/group.py ("gemma" = eager attention and smaller batches; anything else is generic).
Dependencies: steer_true -> steer_xfer; {group, multipos, steer_true, dims, lens} -> 03_events/event_types.py (CPU) -> steer_sweep.
Jobs whose log ends with "rc=0" are skipped, so an interrupted run can be restarted with the same command.
Outputs go to DATA (paths.py); logs to DATA/logs/. Model weights are fetched with huggingface_hub (set HF_HOME to choose the cache)."""
import os, sys, glob, time, subprocess, threading
E = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(E)); from paths import DATA
LOGD = f"{DATA}/logs"; RUN = f"{LOGD}/run.log"; os.makedirs(LOGD, exist_ok=True)
for d in ("dims", "acts", "group", "steer_true", "steer_xfer", "steer_sweep", "multipos", "lens", "lens_full", "handoff", "handoff_patch", "handoff_retrieval", "events"):
    os.makedirs(f"{DATA}/{d}", exist_ok=True)
GPUS = sys.argv[1].split(); SPECS = [s.split(":") for s in sys.argv[2:]]
TASKS = "letter month hour weekday capital2hop city continent color ctxcolor emotion".split()  # largest tasks first
ARITH = ["weekday", "month", "hour", "letter"]
def snap(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.tiktoken"])
lock = threading.Lock()
def log(s):
    with lock: open(RUN, "a").write(f"=== {time.strftime('%H:%M:%S')} {s}\n")
jobs = {}
def add(jid, cwd, cmd, env, deps=(), gpu=True): jobs[jid] = dict(cwd=cwd, cmd=cmd, env=env, deps=list(deps), gpu=gpu)
for mk, name, repo in SPECS:
    p = snap(repo); env = dict(MK=mk, MNAME_OVERRIDE=name, FORCE="1"); pre = []
    for t in TASKS:
        add(f"{name}|steer_true|{t}", f"{E}/02_setup", ["python", "steer_true.py", p, t, name], env)
        add(f"{name}|group|{t}", f"{E}/00_lib", ["python", "group.py", mk, t, "ans", p], env)
        add(f"{name}|multipos|{t}", f"{E}/02_setup", ["python", "multipos.py", p, t, name], env)
        add(f"{name}|dims|{t}", f"{E}/02_setup", ["python", "dims.py", mk, repo, t, name], env)
        add(f"{name}|lens_full|{t}", f"{E}/02_setup", ["python", "lens_full.py", mk, p, name, t], env)
        add(f"{name}|handoff_patch|{t}", f"{E}/04_handoff", ["python", "handoff_patch.py", p, t, name], env)
        add(f"{name}|steer_xfer|{t}", f"{E}/06_transfer", ["python", "steer_xfer.py", p, t, name], env, [f"{name}|steer_true|{t}"])
        pre += [f"{name}|{k}|{t}" for k in ("steer_true", "group", "multipos", "dims")]
    for t in ARITH: add(f"{name}|steer_handoff|{t}", f"{E}/04_handoff", ["python", "steer_handoff.py", p, t, name], env)
    add(f"{name}|lens|all", f"{E}/02_setup", ["python", "lens.py", mk, p, name, ",".join(TASKS)], env); pre.append(f"{name}|lens|all")
    add(f"{name}|events|all", f"{E}/03_events", ["python", "event_types.py"], {}, pre, gpu=False)
    for t in TASKS: add(f"{name}|steer_sweep|{t}", f"{E}/05_onset", ["python", "steer_sweep.py", p, t, name], env, [f"{name}|events|all", f"{name}|steer_xfer|{t}"])
def logf(j): return f"{LOGD}/{j.replace('|', '_')}.log"
done = {j for j in jobs if os.path.exists(logf(j)) and open(logf(j)).read().rstrip().endswith("rc=0")}
running, failed = set(), set()
PRI = {"steer_true": 0, "events": 0, "steer_xfer": 1, "steer_sweep": 2}  # steer_true unlocks the longest chain
order = sorted(jobs, key=lambda j: (PRI.get(j.split("|")[1], 3), list(jobs).index(j)))
def runjob(j, g):
    J = jobs[j]; env = dict(os.environ, OMP_NUM_THREADS="8", **J["env"])
    if g is not None: env["CUDA_VISIBLE_DEVICES"] = g
    log(f"GPU{g} START {j}"); s = time.time()
    with open(logf(j), "w") as f: rc = subprocess.call(J["cmd"], cwd=J["cwd"], env=env, stdout=f, stderr=subprocess.STDOUT)
    open(logf(j), "a").write(f"\nrc={rc}"); log(f"GPU{g} END {j} rc={rc} {time.time() - s:.0f}s")
    with lock: running.discard(j); (done if rc == 0 else failed).add(j)
def blocked(j): return any(d in failed or blocked(d) for d in jobs[j]["deps"])
def ready(j): return j not in done | running | failed and all(d in done for d in jobs[j]["deps"])
log(f"start: {len(jobs)} jobs, {len(done)} already done, GPUs {GPUS}")
while True:
    with lock:
        if not [j for j in order if j not in done | failed and not blocked(j)] and not running: break
        busy = {jobs[j].get("gpu_id") for j in running}
        for j in order:
            if not ready(j): continue
            g = next((x for x in GPUS if x not in busy), None) if jobs[j]["gpu"] else None
            if jobs[j]["gpu"] and g is None: continue
            busy.add(g); jobs[j]["gpu_id"] = g; running.add(j); threading.Thread(target=runjob, args=(j, g), daemon=True).start()
    time.sleep(2)
log(f"ALL DONE: {len(done)} done, failed {sorted(failed)}, blocked {sorted(j for j in jobs if blocked(j))}")
