"""Leave-k-languages-out CV hardness curve (k in {1,2,3}) for the paper's full_tuned
RF (Grambank+WALS). Consistent with fit_models.lolo_folds, a pair is held out when
EITHER endpoint is in the held-out set; train on pairs touching no held-out language.

Design: for each k, form R random disjoint partitions of the 24 source languages
into groups of size k (k=1 is the deterministic LOLO partition). For each
partition, run grouped CV: each fold trains on pairs whose source is outside the
held-out group and predicts pairs whose source is inside it; because the groups
partition the sources, every pair is predicted exactly once per partition, giving
one Pearson rho per partition. The mean over partitions is the reported rho(k);
the 95% CI is the partition-to-partition spread.

Run from the package root:  uv run python experiments/leave_k_languages_out.py
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from experiments import run_experiments as el
from pipeline import fit_models as fm

import os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = ("grambank", "wals")
R = 50          # partitions for k >= 2
KS = (1, 2, 3)
SEED = 42

s = el.load_setup(sources=SOURCES)
pairs = s["pairs"]
y = np.asarray(s["y"], dtype=float)
langs_ids = list(s["langs_ids"])
n = len(langs_ids)
wide = s["wide"].reindex(index=s["langs_ids"])
M, foc, _, feats, idx = el._grambank._one_hot_layout(wide, list(wide.columns))
X = fm.stage2_design(pairs, {g: k for k, g in enumerate(idx)}, M, foc, len(feats))
rfkw = fm.rf_kwargs("full_tuned", SOURCES)
rff = lambda: RandomForestRegressor(n_estimators=200, random_state=42, **rfkw)
src_of = np.array([p[0] for p in pairs])  # source glottocode per directed pair
tgt_of = np.array([p[1] for p in pairs])  # target glottocode per directed pair

rng = np.random.default_rng(SEED)
rows = []
print("Leave-k-languages-out CV (either-endpoint hold-out; GB+WALS, full_tuned RF):")
for k in KS:
    n_part = 1 if k == 1 else R
    rhos = []
    for _ in range(n_part):
        perm = rng.permutation(n)
        groups = [tuple(langs_ids[i] for i in perm[g:g + k]) for g in range(0, n, k)]
        folds = []
        for grp in groups:
            # match fit_models.lolo_folds: a pair is held out if EITHER endpoint
            # is in the group; train on pairs touching no group language.
            in_grp = np.isin(src_of, list(grp)) | np.isin(tgt_of, list(grp))
            folds.append((np.where(~in_grp)[0], np.where(in_grp)[0]))
        pred = fm.cv_predict(X, y, folds, rff)
        rhos.append(fm.pearson(y, pred))
    rhos = np.array(rhos)
    lo, hi = (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))) if len(rhos) > 1 else (np.nan, np.nan)
    rows.append({"k": k, "n_train_langs": n - k, "mean_rho": float(rhos.mean()),
                 "ci_lo": lo, "ci_hi": hi, "n_partitions": len(rhos)})
    print(f"  k={k}: train on {n-k} langs, mean rho = {rhos.mean():.3f}  "
          f"[{lo:.3f}, {hi:.3f}]  over {len(rhos)} partition(s)")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "data", "leave_k_languages_out.csv"), index=False)
print("\nwrote data/leave_k_languages_out.csv")
print("Note: k=1 has a single (deterministic) partition; its cluster-bootstrap CI is [0.51,0.82].")
