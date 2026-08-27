"""Extra inferential statistics that strengthen RQ2-RQ4 of the paper.

Adds paired bootstrap tests (does typology beat simpler/baseline models on the
same pairs?) and cluster / one-sample bootstrap 95% CIs (is each reported number
significantly different from chance / from its rival?). Reuses infer_stats and the
experiment machinery; writes data/inference_summary.csv.

Run from the package root:  uv run python inference_summary.py
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from experiments import run_experiments as el
from pipeline import fit_models as fm
from pipeline import infer_stats as ist

SOURCES = ("grambank", "wals")
B = 2000
SEED = 42
rows = []


def rec(test, estimate, ci_lo, ci_hi, p=np.nan, note=""):
    rows.append({"test": test, "estimate": estimate, "ci_lo": ci_lo,
                 "ci_hi": ci_hi, "p": p, "note": note})
    pstr = f" p={p:.4f}" if np.isfinite(p) else ""
    print(f"  {test:28s} {estimate:+.3f} [{ci_lo:+.3f},{ci_hi:+.3f}]{pstr}  {note}")


# ---------------------------------------------------------------- setup
s = el.load_setup(sources=SOURCES)
pairs = s["pairs"]
y = np.asarray(s["y"], dtype=float)
scripts = s["scripts"]
folds = s["folds"]
blocks = np.array([str(p[0]) for p in pairs])  # cluster by source language

# headline RF LOLO predictions (same construction as run_stats)
wide = s["wide"].reindex(index=s["langs_ids"])
M, foc, _, feats, idx = el._grambank._one_hot_layout(wide, list(wide.columns))
X = fm.stage2_design(pairs, {g: k for k, g in enumerate(idx)}, M, foc, len(feats))
rfkw = fm.rf_kwargs("full_tuned", SOURCES)
rff = lambda: RandomForestRegressor(n_estimators=200, random_state=42, **rfkw)
pred_rf = fm.cv_predict(X, y, folds, rff)

# script-only ridge + interpretable GLM ridge, same pairs/folds
Xscript = np.array([[0.0 if (scripts.get(a) and scripts.get(a) == scripts.get(b)) else 1.0]
                    for a, b in pairs])
pred_script = fm.cv_predict(Xscript, y, folds, lambda: Ridge(alpha=1.0))
Xglm = fm.stage1_design(pairs, s["sim"], s["gene"], s["geo"], scripts)
pred_glm = fm.cv_predict(Xglm, y, folds, lambda: Ridge(alpha=1.0))

print("Paired bootstrap tests (RF vs baselines, same pairs, B=%d):" % B)
for name, base in (("RF - script_only", pred_script), ("RF - glm", pred_glm)):
    t = ist.paired_test(y, pred_rf, base, blocks, metric=fm.pearson, B=B, seed=SEED)
    rec(name, t["estimate"], t["ci_lo"], t["ci_hi"], t["p_value"])

# ---------------------------------------------------------------- subgroup CIs
print("\nWithin/cross-script subgroup CIs (cluster bootstrap):")
same = np.array([bool(scripts.get(a) and scripts.get(a) == scripts.get(b)) for a, b in pairs])
for name, mask in (("RF cross-script", ~same), ("RF same-script", same)):
    ci = ist.bootstrap_ci(y[mask], pred_rf[mask], blocks[mask], metric=fm.pearson, B=B, seed=SEED)
    rec(name, ci["estimate"], ci["ci_lo"], ci["ci_hi"])

# ---------------------------------------------------------------- LOSO/LOFO macro CIs
print("\nOOD macro-rho CIs (one-sample bootstrap over held-out groups):")
rng = np.random.default_rng(SEED)
exp4 = pd.read_csv("data/ood_taxonomy.csv")
for tier in ("LOSO", "LOFO"):
    g = exp4[(exp4.tier == tier) & (exp4.group != "macro")]["pearson"].to_numpy(dtype=float)
    g = g[np.isfinite(g)]
    boots = np.array([np.mean(rng.choice(g, size=len(g), replace=True)) for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rec(f"{tier} macro", float(np.mean(g)), float(lo), float(hi), note=f"{len(g)} groups")

# ---------------------------------------------------------------- Jaccard CI
print("\nFeature-stability CI (bootstrap over fold-pairs):")
all_cols = list(s["wide"].columns)
imp = el._per_fold_importance(s, all_cols)  # DataFrame: feature x fold
fold_cols = list(imp.columns)
tops = {c: set(imp[c].dropna().sort_values(ascending=False).head(25).index) for c in fold_cols}
pair_idx = list(itertools.combinations(range(len(fold_cols)), 2))
jac_vals = np.array([len(tops[fold_cols[i]] & tops[fold_cols[j]]) /
                     len(tops[fold_cols[i]] | tops[fold_cols[j]]) for i, j in pair_idx])
jac_boot = np.array([np.mean(rng.choice(jac_vals, size=len(jac_vals), replace=True)) for _ in range(B)])
lo, hi = np.percentile(jac_boot, [2.5, 97.5])
n_feat, k = len(all_cols), 25
rec("top25 Jaccard", float(np.mean(jac_vals)), float(lo), float(hi),
    note=f"random baseline ~ {k / (2 * n_feat - k):.3f}")

# ---------------------------------------------------------------- Exp5: debias gap CI
print("\nRQ4 debiased gap CI (Marathi vs English on residual bts_T, bootstrap over targets):")
deb = pd.read_csv("outputs/debiased__atlas24/debiased_long__atlas24.csv")
targets = list(s["langs_ids"])
deb_t = deb[deb["target"].isin(targets)]
mat = deb_t.pivot_table(index="source", columns="target", values="bts_T")
ga, gb = mat.loc["mara1378"], mat.loc["stan1293"]
gaps = np.array([ga.iloc[rng.choice(len(mat.columns), size=len(mat.columns), replace=True)].mean() -
                 gb.iloc[rng.choice(len(mat.columns), size=len(mat.columns), replace=True)].mean()
                 for _ in range(B)])
# paired over the same resampled targets:
gaps = np.array([(ga.iloc[idx].mean() - gb.iloc[idx].mean())
                 for idx in (rng.choice(len(mat.columns), size=len(mat.columns), replace=True)
                             for _ in range(B))])
lo, hi = np.percentile(gaps, [2.5, 97.5])
p_gap = 2.0 * min(float(np.mean(gaps <= 0)), float(np.mean(gaps >= 0)))
rec("bias_removed Marathi-English", float(np.mean(gaps)), float(lo), float(hi), p_gap)

# ---------------------------------------------------------------- write
df = pd.DataFrame(rows)
df.to_csv("data/inference_summary.csv", index=False)
print("\nwrote data/inference_summary.csv")
