"""Print the bias-layer B terms (standardized ridge coefficients and per-feature
mean absolute contribution to BTS) for the ATLAS-24 bias layer. Run from the
package root. No RFs are fit, so it is fast."""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from pipeline import fit_models as fm
from pipeline import debias_matrix as dm
from pipeline import data_sources

SOURCES = ("grambank", "wals")
bts, langs_ids, langs, wide, scripts = fm._load_common(0.70, SOURCES)
wiki = data_sources.load_wikipedia_articles()
gene = dm._grambank.genealogical_distance(langs, langs_ids)
geo = dm._grambank.geographic_distance(langs, langs_ids)
pairs = fm.directed_pairs(langs_ids)
bi = bts.set_index(["glottocode_s", "glottocode_t"])
y = np.array([float(bi.loc[(s, t), "bts"]) for s, t in pairs])

bias = dm.fit_bias(pairs, y, scripts, gene, geo, wiki)
ridge = bias["ridge"]
Xb = np.vstack([dm.bias_pair_features(s, t, scripts, gene, geo, wiki,
                                      bias["geo_lo"], bias["geo_hi"]) for s, t in pairs])
scaler = ridge.named_steps["standardscaler"]
coefs = ridge.named_steps["ridge"].coef_
Xs = scaler.transform(Xb)
B = ridge.predict(Xb)

names = ["script_diff", "genealogical_dist", "geo_dist_norm",
         "log1p(resource_s)", "log1p(resource_t)"]
print("feature, std_weight, mean_abs_contrib_BTS")
rows = []
for k, nm in enumerate(names):
    contrib = coefs[k] * Xs[:, k]
    rows.append((nm, float(coefs[k]), float(np.mean(np.abs(contrib)))))
for nm, w, c in sorted(rows, key=lambda r: -abs(r[2])):
    print("%s, %+.4f, %.4f" % (nm, w, c))
print()
print("R2(B,y) = %.3f ; corr(B,y) = %.3f ; n_pairs = %d" %
      (r2_score(y, B), np.corrcoef(y, B)[0, 1], len(y)))
print("mean BTS = %+.3f ; mean B = %+.3f" % (y.mean(), B.mean()))
