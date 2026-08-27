"""Leave-k-out mean rho (k in {2,3}) for the typology tiers, to add as columns to
the RQ1 tiers table. Same fold definition as leave_k_languages_out.py (either-endpoint
hold-out, R random disjoint partitions). Metadata baselines are excluded (they use
a different language count and are already ~0 at LOLO).

Run:  uv run --project <repro package root> python <this file>   (cwd-independent)
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from experiments import run_experiments as el
from pipeline import fit_models as fm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ("grambank", "wals")
R = 25
KS = (2, 3)
SEED = 42

s = el.load_setup(sources=SOURCES)
pairs = s["pairs"]
y = np.asarray(s["y"], dtype=float)
langs_ids = list(s["langs_ids"])
n = len(langs_ids)
wide = s["wide"].reindex(index=s["langs_ids"])
M, foc, _, feats, idx = el._grambank._one_hot_layout(wide, list(wide.columns))
pos = {g: k for k, g in enumerate(idx)}
X2 = fm.stage2_design(pairs, pos, M, foc, len(feats))                          # full / full_tuned
X1 = fm.stage1_design(pairs, s["sim"], s["gene"], s["geo"], s["scripts"])      # glm
Xp = X1[:, [0]]                                                                # proxy (sim only)
src_of = np.array([p[0] for p in pairs])
tgt_of = np.array([p[1] for p in pairs])

tiers = {
    "proxy":      (Xp, lambda: Ridge(alpha=1.0)),
    "glm":        (X1, lambda: Ridge(alpha=1.0)),
    "full":       (X2, lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    "full_tuned": (X2, lambda: RandomForestRegressor(n_estimators=200, random_state=42,
                                                     **fm.rf_kwargs("full_tuned", SOURCES))),
}


def leavek_mean(X, factory, k, rng):
    rhos = []
    for _ in range(R):
        perm = rng.permutation(n)
        groups = [tuple(langs_ids[i] for i in perm[g:g + k]) for g in range(0, n, k)]
        folds = []
        for grp in groups:
            ing = np.isin(src_of, list(grp)) | np.isin(tgt_of, list(grp))
            folds.append((np.where(~ing)[0], np.where(ing)[0]))
        rhos.append(fm.pearson(y, fm.cv_predict(X, y, folds, factory)))
    return float(np.mean(rhos))


rng = np.random.default_rng(SEED)
rows = []
print("Leave-k-out mean rho by typology tier (GB+WALS, R=%d partitions):" % R)
for name, (X, fac) in tiers.items():
    rec = {"tier": name}
    for k in KS:
        rec["leave%dout" % k] = leavek_mean(X, fac, k, rng)
    rows.append(rec)
    print("  %-11s  leave-2-out: %.3f   leave-3-out: %.3f" % (name, rec["leave2out"], rec["leave3out"]))

pd.DataFrame(rows).to_csv(os.path.join(HERE, "data", "leave_k_languages_out_tiers.csv"), index=False)
print("\nwrote data/leave_k_languages_out_tiers.csv")
