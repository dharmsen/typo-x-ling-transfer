# linguistic_validation.py
"""D1: validate the ATLAS BLT matrix on linguistic terms.

Symmetric component: MRM (multiple regression on distance matrices) of transfer
distance on typological / genealogical / script / geographic distance, with
permutation p-values (permute the response matrix's object labels). Partial
Mantel is deliberately NOT used (Type-I bias).
Asymmetric component: OLS of BTS_s->t - BTS_t->s on directed feature diffs (exploratory).
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import numpy as np
import pandas as pd
from pipeline import _grambank
from pipeline import atlas_languages


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, r2


def mrm(predictor_mats, response_mat, B=999, seed=42):
    """MRM with permutation p-values. predictor_mats: list of N x N arrays.
    Returns {coef (per predictor), r2, p (per predictor), B, n}."""
    n = response_mat.shape[0]
    iu = np.triu_indices(n, k=1)
    y = response_mat[iu]
    fin = np.isfinite(y)
    for pm in predictor_mats:
        fin &= np.isfinite(pm[iu])
    cols = [pm[iu][fin] for pm in predictor_mats]
    X = np.column_stack([np.ones(int(fin.sum()))] + cols)
    beta, r2 = _ols(X, y[fin])
    rng = np.random.default_rng(seed)
    null = np.zeros((int(B), len(predictor_mats)))
    for b in range(int(B)):
        perm = rng.permutation(n)
        rp = response_mat[np.ix_(perm, perm)]
        betab, _ = _ols(X, rp[iu][fin])
        null[b] = betab[1:]
    p = np.array([((np.abs(null[:, j]) >= np.abs(beta[j + 1])).sum() + 1) / (B + 1)
                  for j in range(len(predictor_mats))])
    return {"coef": beta[1:], "r2": r2, "p": p, "B": int(B), "n": int(fin.sum())}


def symmetrize_bts(bts_df):
    """bts_df: columns s,t,bts -> Series keyed by (s,t) = mean(s->t, t->s)."""
    f = bts_df.set_index(["s", "t"])["bts"]
    rev = pd.Series({(t, s): v for (s, t), v in f.items()})
    pairs = pd.concat([f, rev])
    pairs = pairs[~pairs.index.duplicated(keep="first")]
    out = {}
    seen = set()
    for (s, t), v in pairs.items():
        key = tuple(sorted((s, t)))
        if key in seen:
            continue
        seen.add(key)
        a = pairs.get((s, t)); b = pairs.get((t, s))
        if pd.notna(a) and pd.notna(b):
            out[(s, t)] = (a + b) / 2
        elif pd.notna(a):
            out[(s, t)] = a
        elif pd.notna(b):
            out[(s, t)] = b
    return pd.Series(out)


def normalize_distance(d):
    d = np.asarray(d, dtype=np.float64)
    lo, hi = np.nanmin(d), np.nanmax(d)
    return (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)


def build_distance_matrices(glottocodes, langs, wide, scripts):
    """Return dict of N x N float arrays: typological, genealogical, geographic, script."""
    sub = _grambank.subset_wide(wide, glottocodes)
    sim = _grambank.simple_similarity(sub).to_numpy()
    typo = _grambank.sim_to_distance(sim)
    gene = _grambank.genealogical_distance(langs, glottocodes).to_numpy()
    geo = _grambank.geographic_distance(langs, glottocodes).to_numpy()
    sc = [scripts.get(g, "") for g in glottocodes]
    n = len(sc)
    script = np.array([[0.0 if sc[i] == sc[j] and sc[i] else 1.0
                        for j in range(n)] for i in range(n)], dtype=np.float64)
    return {"typological": typo, "genealogical": gene, "geographic": geo, "script": script}


def run(bts_path=None, out_dir="outputs", floor=0.70, B=999):
    """D1 symmetric-component battery on the ATLAS matrix.

    Restricts to the coverage-floor-cleared modeling set (for consistency with
    D2/D3) and excludes near-empty varieties like ckb. Writes MRM + per-factor
    Mantel tables to outputs/tables/ AND tracked copies to data/ (outputs/ is
    gitignored). Returns the MRM result dict.
    """
    from pathlib import Path
    out = Path(out_dir); (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parents[1]
    bts = pd.read_csv(bts_path or (here / "data" / "blt_ground_truth.csv"))
    bts = bts.dropna(subset=["bts", "glottocode_s", "glottocode_t"])
    keep = set(atlas_languages.atlas_glottocodes(floor))  # floor-cleared modeling set
    bts = bts[bts["glottocode_s"].isin(keep) & bts["glottocode_t"].isin(keep)]
    langs_ids = sorted(set(bts["glottocode_s"]) | set(bts["glottocode_t"]))
    values, _, langs, _ = _grambank.load_data()
    wide, _ = _grambank.build_wide_matrix(values)
    sc = pd.read_csv(here / "data" / "scripts.csv")
    scripts = dict(zip(sc["glottocode"], sc["script"]))
    dm = build_distance_matrices(langs_ids, langs, wide, scripts)
    sym = symmetrize_bts(bts.rename(columns={"glottocode_s": "s", "glottocode_t": "t",
                                             "bts": "bts"})[["s", "t", "bts"]])
    n = len(langs_ids); idx = {g: k for k, g in enumerate(langs_ids)}
    td = np.full((n, n), np.nan)
    for (s, t), v in sym.items():
        td[idx[s], idx[t]] = td[idx[t], idx[s]] = -v
    td = normalize_distance(td); np.fill_diagonal(td, 0.0)
    # Spec §12: normalize each predictor to [0,1] so MRM coefficients are
    # comparable (geographic is in km, 0-12360). Mantel r below stays raw
    # (it is scale-invariant).
    res = mrm([normalize_distance(dm["typological"]), normalize_distance(dm["script"]),
               normalize_distance(dm["genealogical"]), normalize_distance(dm["geographic"])],
              td, B=B, seed=42)
    mrm_df = pd.DataFrame({"predictor": ["typological", "script", "genealogical", "geographic"],
                           "coef": res["coef"], "perm_p": res["p"]})
    mrm_df.to_csv(out / "tables" / "mrm_symmetric.csv", index=False)
    mrm_df.to_csv(here / "data" / "d1_mrm_symmetric.csv", index=False)
    rows = []
    for name, m in dm.items():
        r = _grambank.mantel_test(td, m, B=B, seed=42)
        rows.append({"factor": name, "r": r["r"], "p": r["p"], "n_pairs": r["n_pairs"]})
    mantel_df = pd.DataFrame(rows)
    mantel_df.to_csv(out / "tables" / "mantel_per_factor.csv", index=False)
    mantel_df.to_csv(here / "data" / "d1_mantel_per_factor.csv", index=False)
    print("D1 (n=%d langs, %d pairs): MRM r2=%.3f" % (n, res["n"], res["r2"]))
    return res
