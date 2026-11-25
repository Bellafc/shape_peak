"""Rebuttal: adjacent-layer change of the answer-class configuration under alternative metrics (CPU only).
Usage: python alt_metrics.py [acts files ...]   (default: every DATA/rebuttal/acts/*.pt)
Output: DATA/rebuttal/metrics/<name>_<task>.csv, one row per layer; change columns are NaN at layer 0.

C_l = K x d matrix of answer-class centroids at layer l (raw residual-stream coordinates, all prompts).

Reference
  procrustes            Eq. (1) of the paper: centre, unit Frobenius norm, optimal rotation + scale. Same as grid_c1_s1_a1.

A. Magnitude only (reviewer 1: "can changes in activation magnitude alone identify similar peaks?")
  mag_scale             |log(||C~_l||_F / ||C~_{l-1}||_F)|, C~ = centred centroids: the overall scale Eq. (1) removes
  mag_norm              |log(mean_x ||h_l(x)|| / mean_x ||h_{l-1}(x)||)|: residual-stream norm growth
  mag_sep               |change of log(||C~_l||_F / (sqrt(K) * mean ||h_l||))|: class spread relative to the residual norm
  mag_classnorm         class-specific magnitude change: || g_l - g_{l-1} - mean(g_l - g_{l-1}) ||_2 with g_l(c) = log ||mu_l^c||
                        (per-class centroid norms, uniform rescaling removed); the only one of the four that is class-specific

C6. Ablation of the three normalisation steps of Eq. (1) (W5 and W3): grid_c{0,1}_s{0,1}_a{0,1}
  c = subtract the mean of the K centroids, s = unit Frobenius norm, a = optimal orthogonal alignment R.
  s=1: disparity of Eq. (1), 1 - (max_R <A R, B>)^2, with R = I when a=0 (optimal scale s>0 kept, so values are in [0, 1]).
  s=0: no scale removal: min_R ||A R - B||_F^2 / ||B||_F^2 (R = I when a=0); a change of overall scale now counts.
  grid_c0_s0_a0 = ||C_l - C_{l-1}||^2 / ||C_l||^2: the raw relative displacement of the centroids in the original coordinates.

C7. Direction change in the original coordinates, no alignment (W3: steering acts on raw coordinates)
  dir_centroid          1 - mean_c cos(mu_l^c - m_l, mu_{l-1}^c - m_{l-1}),  m = mean of the K centroids
  dir_pair              1 - mean_{c<c'} cos(mu_l^c - mu_l^c', mu_{l-1}^c - mu_{l-1}^c'): centroid-difference (steering) directions
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, os, glob, numpy as np, pandas as pd, torch
torch.set_num_threads(8)
OUT = f"{DATA}/rebuttal/metrics"; os.makedirs(OUT, exist_ok=True)

def nuc_cross(A, B):
    """||A^T B||_* for K x d matrices, computed in the span of the rows of A and B (at most 2K dims), as in dims.py."""
    Q, _ = torch.linalg.qr(torch.cat([A, B]).T)
    return float(torch.linalg.svdvals((A @ Q).T @ (B @ Q)).sum())

def grid(A0, B0, c, s, a):
    """A0 = C_{l-1}, B0 = C_l (raw centroids). See module docstring."""
    A, B = (A0 - A0.mean(0), B0 - B0.mean(0)) if c else (A0, B0)
    if s:
        A, B = A / A.norm(), B / B.norm()
        inner = nuc_cross(A, B) if a else max(float((A * B).sum()), 0.0)
        return 1 - inner ** 2
    if a: return (float(A.norm() ** 2 + B.norm() ** 2) - 2 * nuc_cross(A, B)) / float(B.norm() ** 2)
    return float(((A - B) ** 2).sum() / (B ** 2).sum())

def mean_cos(X, Y):
    return float(torch.nn.functional.cosine_similarity(X, Y, dim=1, eps=1e-12).mean())

def per_layer(H, y):
    cls = np.unique(y); K = len(cls); NL = H.shape[1]
    iu = torch.triu_indices(K, K, 1)
    rows, prev = [], None
    for L in range(NL):
        X = H[:, L]
        C = torch.stack([X[y == c].mean(0) for c in cls]); m = C.mean(0); Cc = C - m
        hn = float(X.norm(dim=1).mean()); spread = float(Cc.norm())
        g = torch.log(C.norm(dim=1))
        P = C[iu[0]] - C[iu[1]]
        cur = dict(C=C, Cc=Cc, hn=hn, spread=spread, g=g, P=P, sep=np.log(spread / (np.sqrt(K) * hn)))
        r = dict(layer=L, K=K, resid_norm=hn, class_spread=spread, log_sep=cur["sep"])
        if prev is not None:
            r["procrustes"] = grid(prev["C"], C, 1, 1, 1)
            r["mag_scale"] = abs(np.log(spread / prev["spread"]))
            r["mag_norm"] = abs(np.log(hn / prev["hn"]))
            r["mag_sep"] = abs(cur["sep"] - prev["sep"])
            dg = g - prev["g"]; r["mag_classnorm"] = float((dg - dg.mean()).norm())
            for c in (0, 1):
                for s in (0, 1):
                    for a in (0, 1):
                        r[f"grid_c{c}_s{s}_a{a}"] = grid(prev["C"], C, c, s, a)
            r["dir_centroid"] = 1 - mean_cos(Cc, prev["Cc"])
            r["dir_pair"] = 1 - mean_cos(P, prev["P"])
        rows.append(r); prev = cur
    return pd.DataFrame(rows)

if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(f"{DATA}/rebuttal/acts/*.pt"))
    for f in files:
        a = torch.load(f, weights_only=False); H = a["H"].float(); y = np.asarray(a["ans"])
        name = os.path.basename(f)[:-3]
        d = per_layer(H, y); d["N"] = len(y); d["acc"] = a.get("acc", np.nan)
        d.to_csv(f"{OUT}/{name}.csv", index=False); print("metrics", name, "N =", len(y), "layers =", H.shape[1], flush=True)
