# experiments.py
"""Targeted experiments for the paper's research questions.

Exp1 — Within-script typology isolation: does typology predict cross-lingual
        transfer BEYOND orthography? Decompose held-out (LOLO) prediction
        accuracy into same-script vs cross-script vs Latin-Latin pairs. Script
        alone is constant within same-script pairs, so any within-script signal
        is genuinely typological (genealogical/geographic/typological), not
        orthographic.

Exp2 — Linguistic-subsystem ablation: which typological subsystem carries the
        transfer signal? (a) keyword-classify Grambank features into linguistic
        domains, restrict the RF to each domain, score LOLO; (b) leave-one-
        domain-out; (c) domain enrichment of the top permutation-importance
        features vs their base rate.

Writes committed result tables to data/*.csv; also prints them.
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from pipeline import _grambank
from pipeline import checkpoint as cp
from pipeline import fit_models as fm
from pipeline import paths

HERE = Path(__file__).parents[1]
GB_PARAMS = paths.grambank_data_dir() / "parameters.csv"


def _commit_table(stem, df):
    """Write df to data/<stem>.csv (committed result tables). Returns the path."""
    out = HERE / "data" / stem
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_setup(sources=("grambank", "wals"), floor=0.70, mode="gb"):
    bts, langs_ids, langs, wide, scripts = fm._load_common(floor, sources, mode)
    sim = _grambank.simple_similarity(_grambank.subset_wide(wide, langs_ids))
    gene = _grambank.genealogical_distance(langs, langs_ids)
    geo = _grambank.geographic_distance(langs, langs_ids)
    pairs = fm.directed_pairs(langs_ids)
    bts_idx = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bts_idx.loc[(s, t), "bts"]) for s, t in pairs])
    folds = fm.lolo_folds(pairs, langs_ids)
    return dict(langs_ids=langs_ids, wide=wide, scripts=scripts, pairs=pairs,
                y=y, folds=folds, sim=sim, gene=gene, geo=geo, sources=sources)


def rf_predict(setup, sources):
    """Held-out (LOLO) RandomForest predictions over all directed pairs."""
    wide = setup["wide"].reindex(index=setup["langs_ids"])
    M, foc, _, feats, idx = _grambank._one_hot_layout(wide, list(wide.columns))
    pos = {g: k for k, g in enumerate(idx)}
    X = fm.stage2_design(setup["pairs"], pos, M, foc, len(feats))
    factory = lambda: RandomForestRegressor(
        n_estimators=200, random_state=42, **fm.rf_kwargs("full_tuned", sources, "gb"))
    return fm.cv_predict(X, setup["y"], setup["folds"], factory)


def rf_predict_on_features(setup, feat_subset, max_features=None):
    """LOLO RF predictions using only `feat_subset` columns of the wide matrix."""
    wide = setup["wide"].reindex(index=setup["langs_ids"])
    M, foc, _, feats, idx = _grambank._one_hot_layout(wide, feat_subset)
    pos = {g: k for k, g in enumerate(idx)}
    X = fm.stage2_design(setup["pairs"], pos, M, foc, len(feats))
    kw = {"n_estimators": 200, "random_state": 42}
    if max_features:
        kw["max_features"] = max_features
    factory = lambda: RandomForestRegressor(**kw)
    return fm.cv_predict(X, setup["y"], setup["folds"], factory)


# ---------------------------------------------------------------------------
# Experiment 1: within-script typology isolation
# ---------------------------------------------------------------------------

def experiment1():
    print("=" * 78)
    print("EXP 1 — Within-script typology isolation (does typology beat orthography?)")
    print("=" * 78)
    out = []
    for sources in [("grambank", "wals"), ("grambank",)]:
        s = load_setup(sources=sources)
        pairs, y, scripts = s["pairs"], s["y"], s["scripts"]
        same = np.array([bool(scripts.get(a) and scripts.get(a) == scripts.get(b))
                         for a, b in pairs])
        latin = np.array([scripts.get(a) == "Latin" and scripts.get(b) == "Latin"
                          for a, b in pairs])
        n_same, n_cross, n_lat = int(same.sum()), int((~same).sum()), int(latin.sum())

        # models
        Xscript = np.array([[0.0 if (scripts.get(a) and scripts.get(a) == scripts.get(b)) else 1.0]
                            for a, b in pairs])
        pred_script = fm.cv_predict(Xscript, y, s["folds"], lambda: Ridge(alpha=1.0))
        Xglm = fm.stage1_design(pairs, s["sim"], s["gene"], s["geo"], scripts)  # sim,script,gene,geo
        pred_glm = fm.cv_predict(Xglm, y, s["folds"], lambda: Ridge(alpha=1.0))
        pred_glm_typo = fm.cv_predict(Xglm[:, [0, 2, 3]], y, s["folds"],  # drop script col
                                      lambda: Ridge(alpha=1.0))
        pred_rf = rf_predict(s, sources)

        def row(name, pred):
            return {
                "model": name,
                "overall": fm.pearson(y, pred),
                "same_script": fm.pearson(y[same], pred[same]),
                "cross_script": fm.pearson(y[~same], pred[~same]),
                "latin_latin": fm.pearson(y[latin], pred[latin]),
            }

        src = "+".join(sources)
        print(f"\n-- sources={src}  (n_same={n_same}, n_cross={n_cross}, n_latin={n_lat}) --")
        df = pd.DataFrame([
            row("script_only", pred_script),
            row("glm (sim,script,gene,geo)", pred_glm),
            row("glm_typo (sim,gene,geo) NO script", pred_glm_typo),
            row("rf_full_tuned", pred_rf),
        ])
        out.append((src, df, (n_same, n_cross, n_lat)))
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        _commit_table("within_script__%s.csv" % src.replace("+", "_"), df)
    return out


# ---------------------------------------------------------------------------
# Experiment 2: linguistic-subsystem ablation
# ---------------------------------------------------------------------------

def classify_domain(name):
    """Transparent first-match keyword classifier -> linguistic domain."""
    n = name.lower()
    if any(k in n for k in ["interrogative", "polar interrogation", "question"]):
        return "Interrogatives"
    if "negat" in n:
        return "Negation"
    if any(k in n for k in ["order of", "verb-initial", "verb-medial", "verb-final",
                            "unmarked order", "unmarked constituent order",
                            "special word order", "order of core", "order of s and v"]):
        return "Word Order"
    if "possess" in n:
        return "Possession"
    if any(k in n for k in ["numeral system", "vigesimal", "quinary", "decimal numeral",
                            "body-part tally", "tallying"]):
        return "Numerals"
    if "article" in n:
        return "Nominal: Articles"
    if any(k in n for k in ["gender", "noun class", "classifier"]):
        return "Nominal: Classification"
    if any(k in n for k in ["singular", "dual", "plural", "paucal", "trial",
                            "number marker", "number markers", "suppletive for number",
                            "noun number", "diminutive", "augmentative"]):
        return "Nominal: Number"
    if any(k in n for k in ["demonstrative", "pronoun", "inclusive and exclusive",
                            "logophoric", "reflexive", "reciprocal", "politeness"]):
        return "Pronouns/Demonstratives"
    if any(k in n for k in ["morphological case", "preposition", "postposition",
                            "adposition", "alignment", "flagging", "core participant",
                            "core argument", "ditransitive"]):
        return "Case/Adpositions/Alignment"
    if any(k in n for k in ["relative clause", "complementizer", "clause chaining",
                            "switch reference", "simultaneous and sequential",
                            "serial verb", "conjunction", "comparative"]):
        return "Clause Combining"
    if any(k in n for k in ["deriving an", "derivation"]):
        return "Nominal Derivation"
    if any(k in n for k in ["verb", "tense", "aspect", "mood", "auxiliary",
                            "applicative", "passive", "antipassive", "inverse",
                            "conjugation", "suppletive", "reduplicate", "argument",
                            "index", "copula", "existential", "causative", "pro-drop",
                            "null anaphora", "ideophone", "posture verb"]):
        return "Verbal Categories"
    if any(k in n for k in ["universal quantifier", "quantifier", "count/mass"]):
        return "Quantifiers"
    if any(k in n for k in ["evidential", "evidence"]):
        return "Evidentiality"
    return "Other"


def experiment2():
    print("\n" + "=" * 78)
    print("EXP 2 — Linguistic-subsystem ablation (which subsystem predicts transfer?)")
    print("=" * 78)
    s = load_setup(sources=("grambank",))  # GB-only: clean typology
    wide = s["wide"]
    gb_cols = [c for c in wide.columns if not c.startswith("WALS_")]
    with open(GB_PARAMS, newline="") as fh:
        names = {r["ID"]: r["Name"] for r in csv.DictReader(fh)}
    domain = {c: classify_domain(names.get(c, c)) for c in gb_cols}

    # domain inventory
    dom_count = Counter(domain.values())
    print("\nDomain inventory (GB features in the wide matrix):")
    for d, n in dom_count.most_common():
        print(f"  {n:3d}  {d}")

    # (a) per-domain RF LOLO + (b) leave-one-domain-out
    y = s["y"]
    all_pred = rf_predict_on_features(s, gb_cols, max_features=0.5)
    base_rho = fm.pearson(y, all_pred)
    rows = []
    for d in dom_count:
        cols_d = [c for c in gb_cols if domain[c] == d]
        if len(cols_d) < 5:
            continue
        pred_d = rf_predict_on_features(s, cols_d, max_features=0.5)
        # leave-one-domain-out
        cols_loo = [c for c in gb_cols if domain[c] != d]
        pred_loo = rf_predict_on_features(s, cols_loo, max_features=0.5)
        rows.append({
            "domain": d, "n_feat": len(cols_d),
            "per_domain_rho": fm.pearson(y, pred_d),
            "loo_rho": fm.pearson(y, pred_loo),
            "loo_drop": base_rho - fm.pearson(y, pred_loo),
        })
    df_abl = pd.DataFrame(rows).sort_values("per_domain_rho", ascending=False)
    print(f"\nAll-features (GB) LOLO rho = {base_rho:.3f}")
    print("\nPer-domain RF (LOLO Pearson) and leave-one-domain-out:")
    print(df_abl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    _commit_table("domain_ablation.csv", df_abl)

    # (c) domain enrichment of top permutation-importance features
    rank = pd.read_csv(HERE / "data" / "d2_feature_importance.csv")
    rank = rank[rank["feature_id"].isin(gb_cols)].sort_values("importance", ascending=False)
    top = rank.head(25)
    base_rate = {d: dom_count[d] / len(gb_cols) for d in dom_count}
    top_rate = Counter(domain[c] for c in top["feature_id"])
    print("\nDomain enrichment of top-25 permutation-importance features:")
    print(f"{'domain':32s} {'top25':>6} {'base%':>6} {'top%':>6} {'lift':>6}")
    for d, n in sorted(top_rate.items(), key=lambda kv: -kv[1]):
        bp = 100 * base_rate[d]
        tp = 100 * n / 25
        print(f"{d:32s} {n:6d} {bp:6.1f} {tp:6.1f} {tp / bp:6.2f}")
    print("\nTop-10 features with domains:")
    for _, r in rank.head(10).iterrows():
        c = r["feature_id"]
        print(f"  {c}  [{domain[c]:24s}]  imp={r['importance']:+.3f}  {names.get(c,'')[:60]}")

    return df_abl


# ---------------------------------------------------------------------------
# Experiment 3: feature-distribution sweep + fold-stability
# ---------------------------------------------------------------------------

def _fit_pred(X, y, tr, te, factory):
    m = factory()
    m.fit(X[tr], y[tr])
    return m.predict(X[te])


def _per_fold_importance(setup, feat_cols, model_factory=None, seed=42):
    """Per LOLO fold, per-feature importance = drop in THAT fold's held-out Pearson
    when the feature's one-hot columns are shuffled within the fold's training set.
    Returns DataFrame[feature x fold] (fold columns named f0, f1, ...)."""
    model_factory = model_factory or (lambda: Ridge(alpha=1.0))
    wide = setup["wide"].reindex(index=setup["langs_ids"])
    M, foc, _, feats, idx = _grambank._one_hot_layout(wide, feat_cols)
    pos = {g: k for k, g in enumerate(idx)}
    X = fm.stage2_design(setup["pairs"], pos, M, foc, len(feats))
    y = np.asarray(setup["y"], dtype=float)
    rng = np.random.default_rng(seed)
    n_codes = len(foc)
    cols_by_feat = {f: list(np.where(foc == f)[0]) for f in range(len(feats))}
    out = {feats[f]: [] for f in range(len(feats))}
    for tr, te in setup["folds"]:
        base = fm.pearson(y[te], _fit_pred(X, y, tr, te, model_factory))
        for f in range(len(feats)):
            if not cols_by_feat[f]:
                out[feats[f]].append(float("nan"))
                continue
            Xp = X.copy()
            for col in cols_by_feat[f]:
                Xp[tr, col] = X[tr, col][rng.permutation(len(tr))]
                Xp[tr, n_codes + col] = X[tr, n_codes + col][rng.permutation(len(tr))]
            Xp[tr, 2 * n_codes + f] = X[tr, 2 * n_codes + f][rng.permutation(len(tr))]
            dropped = fm.pearson(y[te], _fit_pred(Xp, y, tr, te, model_factory))
            out[feats[f]].append(base - dropped)
    return pd.DataFrame(out, index=["f%d" % i for i in range(len(setup["folds"]))]).T


def experiment3(ks=(1, 5, 10, 25, 50, 100, 200)):
    """RQ3: is transfer-relevant typology distributed across many features?
    (a) cumulative feature-count sweep (rho vs k); (b) fold-stability of the
    permutation-importance ranking. Runs on the combined Grambank+WALS feature
    space (the paper's default source) for consistency with the other RQs.

    The model split is intentional: the feature-count sweep (a) uses the RF
    (via rf_predict_on_features), while the fold-stability (b) is measured on
    the Ridge permutation-importance ranking -- Ridge is deterministic (so RF
    fit-stochasticity can't inflate the instability estimate) and fast."""
    print("\n" + "=" * 78)
    print("EXP 3 — Feature-distribution sweep + fold-stability (Grambank+WALS)")
    print("=" * 78)
    s = load_setup(sources=("grambank", "wals"))
    y = s["y"]
    all_cols = list(s["wide"].columns)
    rank = pd.read_csv(HERE / "data" / "d2_feature_importance__grambank_wals.csv")
    rank = rank[rank["feature_id"].isin(all_cols)].sort_values("importance", ascending=False)
    ordered = rank["feature_id"].tolist()

    base = fm.pearson(y, rf_predict_on_features(s, all_cols, max_features=0.5))
    rows = []
    for k in ks:
        cols = ordered[:k]
        if not cols:
            continue
        rows.append({"k": k, "lolo_pearson": fm.pearson(y, rf_predict_on_features(s, cols, max_features=0.5))})
    rows.append({"k": len(all_cols), "lolo_pearson": base})
    sweep = pd.DataFrame(rows)
    _commit_table("feature_count_sweep.csv", sweep)
    print("feature-count sweep (LOLO Pearson, Grambank+WALS):")
    print(sweep.to_string(index=False, float_format=lambda v: "%.3f" % v))

    imp = _per_fold_importance(s, all_cols)
    from scipy.stats import spearmanr
    vecs = [imp[c].to_numpy(dtype=float) for c in imp.columns]
    spears = [spearmanr(vecs[i], vecs[j]).statistic for i in range(len(vecs)) for j in range(i + 1, len(vecs))
              if np.isfinite(vecs[i]).all() and np.isfinite(vecs[j]).all()]
    tops = [set(imp[c].dropna().sort_values(ascending=False).head(25).index) for c in imp.columns]
    jacc = [len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]] if len(tops) > 1 else [1.0]
    stab = pd.DataFrame([{
        "mean_fold_pair_spearman": float(np.nanmean(spears)) if spears else float("nan"),
        "mean_top25_jaccard": float(np.mean(jacc)) if jacc else float("nan"),
    }])
    _commit_table("importance_stability.csv", stab)
    print("fold-stability: mean pairwise Spearman %.3f | mean top-25 Jaccard %.3f"
          % (stab["mean_fold_pair_spearman"][0], stab["mean_top25_jaccard"][0]))
    return sweep, stab


# ---------------------------------------------------------------------------
# Experiment 3b: leak-free top-k ablation (E4 selection-optimism control)
# ---------------------------------------------------------------------------

def _nested_sub_setup(setup, fold_i):
    """Setup restricted to fold i's TRAINING pairs: every language, pair and
    inner fold excludes fold i's held-out language entirely (the ranking must
    be recomputed entirely inside each outer training set, not averaged over
    the other outer folds, which still contain the held-out language in their
    train and test sets)."""
    tr, te = setup["folds"][fold_i]
    held = setup["langs_ids"][fold_i]
    assert all(held in p for p in (setup["pairs"][j] for j in te)), \
        "fold i's test pairs must all contain langs_ids[i]"
    sub_langs = [g for g in setup["langs_ids"] if g != held]
    sub_pairs = [setup["pairs"][j] for j in tr]
    return {"langs_ids": sub_langs, "wide": setup["wide"], "pairs": sub_pairs,
            "y": np.asarray(setup["y"], dtype=float)[tr],
            "folds": fm.lolo_folds(sub_pairs, sub_langs)}


def _nested_fold_ranking(setup, fold_i, feat_cols, model_factory=None, seed=42):
    """Feature ranking for LOLO fold i, computed by inner LOLO permutation
    importance entirely inside fold i's training pairs: no pair containing the
    held-out language touches the ranking at any stage. NaN (feature has no
    one-hot columns in the inner folds) sorts last."""
    sub = _nested_sub_setup(setup, fold_i)
    imp = _per_fold_importance(sub, feat_cols, model_factory, seed)
    return imp.mean(axis=1).sort_values(ascending=False, na_position="last")


def _leakfree_topk_lolo(setup, rankings, k, factory=None):
    """LOLO predictions where each fold's model trains on that fold's OWN top-k
    features (rankings[i], from _nested_fold_ranking): the held-out language
    touches neither the fit nor the feature ranking. Factory defaults to the
    E4 sweep RF (mf=0.5)."""
    factory = factory or (lambda: RandomForestRegressor(
        n_estimators=200, random_state=42, max_features=0.5))
    wide = setup["wide"].reindex(index=setup["langs_ids"])
    y = np.asarray(setup["y"], dtype=float)
    preds = np.full(len(y), np.nan)
    for i, (tr, te) in enumerate(setup["folds"]):
        top = set(rankings[i].head(k).index)
        cols = [c for c in wide.columns if c in top]   # canonical wide order
        M, foc, _, feats, idx = _grambank._one_hot_layout(wide, cols)
        pos = {g: j for j, g in enumerate(idx)}
        X = fm.stage2_design(setup["pairs"], pos, M, foc, len(feats))
        m = factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])   # last-write-wins, matching fm.cv_predict
    return preds


def _seed_stability(setup, rankings, seeds=(42, 43, 44), k=200, factory=None):
    """One row of seed-stability statistics: LOLO rho of the full feature set
    and of the leak-free top-k pipeline, each mean +- std over `seeds`. The
    factory maps a seed to a fresh unfitted model constructor."""
    rf = factory or (lambda seed: lambda: RandomForestRegressor(
        n_estimators=200, random_state=seed, max_features=0.5))
    M, foc, _, feats, idx = _grambank._one_hot_layout(
        setup["wide"].reindex(index=setup["langs_ids"]),
        list(setup["wide"].columns))
    X_full = fm.stage2_design(setup["pairs"], {g: j for j, g in enumerate(idx)},
                              M, foc, len(feats))
    full = [fm.pearson(setup["y"],
                       fm.cv_predict(X_full, setup["y"], setup["folds"], rf(sd)))
            for sd in seeds]
    leak = [fm.pearson(setup["y"],
                       _leakfree_topk_lolo(setup, rankings, k, factory=rf(sd)))
            for sd in seeds]
    return {"full_mean": float(np.mean(full)), "full_std": float(np.std(full)),
            "leakfree_mean": float(np.mean(leak)), "leakfree_std": float(np.std(leak)),
            "k": k, "n_seeds": len(seeds)}


def experiment3b(ks=(1, 5, 10, 25, 50, 100, 200), ckpt_dir=None):
    """RQ3 ablation: is E4's top-k peak (rho 0.742 at k=200 vs 0.705 full) real,
    or selection optimism? The E4 sweep orders features by the global d2 ranking,
    whose importance is the drop in the FULL LOLO metric -- the same folds the
    sweep is re-evaluated on -- so every k<387 point is optimistically biased.
    Here each fold selects its own top-k by NESTED importance: an inner LOLO
    over that fold's training pairs only, so the held-out language touches
    neither the fit nor the ranking at any stage. Ranking uses the Ridge
    permutation importance (same protocol as the fold-stability analysis:
    deterministic, fast); evaluation uses the same RF (mf=0.5) as the sweep.
    With `ckpt_dir`, per-fold nested rankings are checkpointed (JSONL) and the
    run can be interrupted and resumed: completed folds are skipped on restart."""
    print("\n" + "=" * 78)
    print("EXP 3b — Leak-free top-k feature sweep (Grambank+WALS)")
    print("=" * 78)
    s = load_setup(sources=("grambank", "wals"))
    all_cols = list(s["wide"].columns)

    ck = Path(ckpt_dir or (HERE / "outputs" / "nested")) / "t1_fold_rankings.jsonl"
    done = {r["fold"]: pd.Series(dict(r["ranking"])) for r in cp.load_jsonl(ck)}
    rankings = []
    for i in range(len(s["folds"])):
        if i in done:
            print("  fold %2d/%d: ranking loaded from checkpoint" % (i + 1, len(s["folds"])))
            rankings.append(done[i])
            continue
        print("  fold %2d/%d: nested inner LOLO importance (%d features)..."
              % (i + 1, len(s["folds"]), len(all_cols)))
        r = _nested_fold_ranking(s, i, all_cols)
        rankings.append(r)
        cp.append_jsonl(ck, {"fold": i,
                             "ranking": [[feat, float(v)] for feat, v in r.items()]})

    rows = [{"k": k, "lolo_pearson": fm.pearson(s["y"], _leakfree_topk_lolo(s, rankings, k))}
            for k in ks]
    rows.append({"k": len(all_cols),
                 "lolo_pearson": fm.pearson(s["y"], rf_predict_on_features(s, all_cols, max_features=0.5))})
    sweep = pd.DataFrame(rows)
    _commit_table("leakfree_topk_sweep.csv", sweep)

    global_sweep = pd.read_csv(HERE / "data" / "feature_count_sweep.csv")
    merged = (sweep.merge(global_sweep, on="k", suffixes=("_leakfree", "_globalrank"))
              .rename(columns={"lolo_pearson_leakfree": "leakfree",
                               "lolo_pearson_globalrank": "globalrank"}))
    print("top-k sweep: leak-free vs committed global-rank (LOLO Pearson):")
    print(merged[["k", "leakfree", "globalrank"]].to_string(
        index=False, float_format=lambda v: "%.3f" % v))

    stab = _seed_stability(s, rankings)
    _commit_table("leakfree_seed_stability.csv", pd.DataFrame([stab]))
    print("seed stability (seeds 42,43,44): full-387 rho %.3f +- %.3f | leak-free k=200 rho %.3f +- %.3f"
          % (stab["full_mean"], stab["full_std"],
             stab["leakfree_mean"], stab["leakfree_std"]))
    return sweep


# ---------------------------------------------------------------------------
# Experiment 4: OOD taxonomy (LOLO / LOSO / LOFO)
# ---------------------------------------------------------------------------

def experiment4():
    """RQ2 (OOD): generalization ladder — LOLO (in-distribution), LOSO/LOFO
    (within-ATLAS shift), true extrapolation (non-ATLAS, scope-only). Reuses
    debias_matrix.grouped_cv_pearson for the grouped folds."""
    from pipeline import debias_matrix as dm
    from pipeline import data_sources
    print("\n" + "=" * 78)
    print("EXP 4 — OOD taxonomy (LOLO / LOSO / LOFO)")
    print("=" * 78)
    s = load_setup(sources=("grambank", "wals"))
    wide = s["wide"].reindex(index=s["langs_ids"])
    M, foc, _, feats, idx = _grambank._one_hot_layout(wide, list(wide.columns))
    pos = {g: k for k, g in enumerate(idx)}
    X = fm.stage2_design(s["pairs"], pos, M, foc, len(feats))
    y, pairs, langs_ids, scripts = s["y"], s["pairs"], s["langs_ids"], s["scripts"]
    rfkw = fm.rf_kwargs("full_tuned", ("grambank", "wals"))
    factory = lambda: RandomForestRegressor(
        n_estimators=200, random_state=42, **rfkw)

    lolo_pred = fm.cv_predict(X, y, s["folds"], factory)
    lolo_rho = fm.pearson(y, lolo_pred)
    rows = [{"tier": "LOLO", "group": "macro", "pearson": lolo_rho}]

    family = data_sources.wals_family_genus()
    script_of = {g: scripts[g] for g in langs_ids if scripts.get(g)}
    family_of = {g: family[g][0] for g in langs_ids if g in family}
    for axis, group_of in (("LOSO", script_of), ("LOFO", family_of)):
        per_group = dm.grouped_cv_pearson(X, y, pairs, group_of, factory)
        for g, v in per_group.items():
            rows.append({"tier": axis, "group": g, "pearson": v})
        macro = float(np.nanmean([v for v in per_group.values() if np.isfinite(v)])) if per_group else float("nan")
        rows.append({"tier": axis, "group": "macro", "pearson": macro})

    rows.append({"tier": "TRUE_EXTRAP", "group": "scope",
                 "pearson": float("nan")})  # non-ATLAS: no ground truth -> scope
    df = pd.DataFrame(rows)
    _commit_table("ood_taxonomy.csv", df)
    print(df.to_string(index=False, float_format=lambda v: "%.3f" % v))
    return df


# ---------------------------------------------------------------------------
# Experiment 5: RQ4 — best source under three operationalizations
# ---------------------------------------------------------------------------

def _bias_removed_onscale_scores(deb_df, bias_layer_df, sources, targets):
    """On-scale bias_removed scores for RQ4: mean (bts_T + bts_B) per source over
    `targets`, joining the debiased long table with the ATLAS bias layer. Pure
    (no I/O) -- experiment5 reads the CSVs and passes them in. Returns
    {source: score}, restricted to `sources` that appear in the joined table."""
    merged = (deb_df[["source", "target", "bts_T"]]
              .merge(bias_layer_df[["source", "target", "bts_B"]], on=["source", "target"]))
    merged["bts_T_onscale"] = merged["bts_T"] + merged["bts_B"]
    score = (merged[merged["target"].isin(targets)]
             .groupby("source")["bts_T_onscale"].mean())
    return {g: float(score.loc[g]) for g in sources if g in score.index}


def experiment5(sources=None, targets=None, top_n=10):
    """RQ4: is English the best source — empirically (ground truth) and typologically
    (three operationalizations), over a FIXED candidate source set and target set.
    1. bias_removed: mean bts_T (debiased residual) over targets.
    2. centrality: mean Grambank similarity to the targets.
    3. ood_generalizer: mean transfer to held-out-family (LOFO-style) targets.
    `sources`/`targets` default to the full ATLAS-24 modeling set."""
    from pipeline import data_sources
    print("\n" + "=" * 78)
    print("EXP 5 — RQ4: best source (empirical + three typological operationalizations)")
    print("=" * 78)
    s = load_setup(sources=("grambank", "wals"))
    langs_ids = s["langs_ids"]
    sources = list(sources) if sources else list(langs_ids)
    targets = list(targets) if targets else list(langs_ids)
    wide = s["wide"]

    # 1. bias_removed: mean transfer to the TARGETS from the debiased residual if
    #    available, else the ATLAS ground-truth bts (raw, not residual).
    deb_path = HERE / "outputs" / "debiased__atlas24" / "debiased_long__atlas24.csv"
    if deb_path.exists():
        deb = pd.read_csv(deb_path)
        score = deb[deb["target"].isin(targets)].groupby("source")["bts_T"].mean()
        print("bias_removed source: debiased residual bts_T (debiased_long CSV)")
    else:
        gt0 = pd.read_csv(HERE / "data" / "blt_ground_truth.csv")
        score = gt0[gt0["glottocode_t"].isin(targets)].groupby("glottocode_s")["bts"].mean()
        print("bias_removed source: ATLAS ground-truth bts (debiased CSV absent — raw, not residual)")
    bias_removed = [(g, float(score.loc[g])) for g in sources if g in score.index]

    # 2. centrality: mean Grambank similarity to the targets (exclude self-similarity).
    want = [g for g in set(sources) | set(targets) if g in wide.index]
    sim = _grambank.simple_similarity(_grambank.subset_wide(wide, want))
    tgts = [g for g in targets if g in sim.columns]
    centrality = []
    for g in sources:
        if g not in sim.index:
            continue
        tg = [t for t in tgts if t != g]
        if tg:
            centrality.append((g, float(sim.loc[g, tg].mean())))

    # 3. ood_generalizer: mean transfer to held-out-family targets (family of target != family of source).
    family = data_sources.wals_family_genus()
    fam_of = {g: family[g][0] for g in targets if g in family}
    gt = pd.read_csv(HERE / "data" / "blt_ground_truth.csv")
    ood = []
    for g in sources:
        gfam = family.get(g, (None, ""))[0]
        ood_targets = [t for t in targets if t in fam_of and fam_of[t] != gfam]
        if ood_targets:
            sub = gt[(gt["glottocode_s"] == g) & (gt["glottocode_t"].isin(ood_targets))]
            ood.append((g, float(sub["bts"].mean()) if len(sub) else float("nan")))

    def _rank(name, pairs):
        pairs = [(g, v) for g, v in pairs if np.isfinite(v)]
        pairs = sorted(pairs, key=lambda kv: kv[1], reverse=True)[:top_n]
        return [{"operationalization": name, "rank": i + 1, "source": g, "score": v}
                for i, (g, v) in enumerate(pairs)]

    df = pd.DataFrame(_rank("bias_removed", bias_removed)
                      + _rank("centrality", centrality)
                      + _rank("ood_generalizer", ood))
    _commit_table("best_source_rankings.csv", df)
    print(df.to_string(index=False, float_format=lambda v: "%.3f" % v))

    # On-scale bias_removed: bts_T + bts_B (back on the BTS scale; ATLAS-24 only,
    # where the bias ridge B is defined). Stored in a SEPARATE csv so the legacy
    # off-scale residual ranking (best_source_rankings.csv) is preserved.
    bl_path = HERE / "outputs" / "debiased__atlas24" / "bias_layer__atlas24.csv"
    if deb_path.exists() and bl_path.exists():
        bl = pd.read_csv(bl_path)
        os_scores = _bias_removed_onscale_scores(deb, bl, sources, targets)
        bias_removed_os = [(g, v) for g, v in os_scores.items()]
        df_os = pd.DataFrame(_rank("bias_removed_onscale", bias_removed_os)
                             + _rank("centrality", centrality)
                             + _rank("ood_generalizer", ood))
        _commit_table("best_source_rankings_onscale.csv", df_os)
        print("\nbias_removed_onscale source: bts_T + bts_B (on BTS scale; ATLAS-24)")
        print(df_os[df_os["operationalization"] == "bias_removed_onscale"]
              .to_string(index=False, float_format=lambda v: "%.3f" % v))
    else:
        print("\nbias_removed_onscale: skipped (debiased_long/bias_layer CSV absent)")
    return df


# ---------------------------------------------------------------------------
# Headline inferential stats (RQ1/RQ2)
# ---------------------------------------------------------------------------

def run_stats(sources=("grambank", "wals"), B_null=200, B_boot=1000):
    """Headline inferential stats for RQ1/RQ2: permutation null + cluster bootstrap
    CI on the LOLO Pearson + true R^2, for the full_tuned RF. NOTE: B_null RF
    refits across all LOLO folds is slow (~B_null minutes); lower B_null for a
    quick check."""
    from pipeline import infer_stats as ist
    print("\n" + "=" * 78)
    print("STATS — permutation null + cluster bootstrap CIs (LOLO)")
    print("=" * 78)
    s = load_setup(sources=sources)
    wide = s["wide"].reindex(index=s["langs_ids"])
    M, foc, _, feats, idx = _grambank._one_hot_layout(wide, list(wide.columns))
    X = fm.stage2_design(s["pairs"], {g: k for k, g in enumerate(idx)}, M, foc, len(feats))
    y, pairs, folds = s["y"], s["pairs"], s["folds"]
    blocks = np.array([str(p[0]) for p in pairs])  # cluster by source language
    rfkw = fm.rf_kwargs("full_tuned", sources)
    rff = lambda: RandomForestRegressor(n_estimators=200, random_state=42, **rfkw)

    pred = fm.cv_predict(X, y, folds, rff)
    rho = fm.pearson(y, pred)
    null = ist.permutation_null(X, y, folds, rff, B=B_null, seed=42)
    ci_rho = ist.bootstrap_ci(y, pred, blocks, metric=fm.pearson, B=B_boot, seed=42)
    ci_r2 = ist.bootstrap_ci(y, pred, blocks, metric=fm.r2, B=B_boot, seed=42)
    rows = [
        {"metric": "lolo_pearson", "estimate": rho, "p_above_chance": null["p_value"],
         "ci_lo": ci_rho["ci_lo"], "ci_hi": ci_rho["ci_hi"]},
        {"metric": "lolo_true_r2", "estimate": fm.r2(y, pred), "p_above_chance": None,
         "ci_lo": ci_r2["ci_lo"], "ci_hi": ci_r2["ci_hi"]},
    ]
    df = pd.DataFrame(rows)
    _commit_table("stats__lolo.csv", df)
    print(df.to_string(index=False, float_format=lambda v: "%.3f" % v))
    print("permutation null mean rho = %.3f (B=%d)" % (null["null_mean"], null["B"]))
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Experiments (Exp1-5 + stats).")
    ap.add_argument("--exp", choices=["1", "2", "3", "3b", "4", "5", "stats", "all"], default="all")
    a = ap.parse_args()
    # NOTE: experiments are parameterized via argparse + load_setup defaults; the
    # configs/*.yaml files document the canonical params (full config.py-wiring is
    # deferred — the recommender in Plan 1 is the config-driven reference).
    if a.exp in ("1", "all"): experiment1()
    if a.exp in ("2", "all"): experiment2()
    if a.exp in ("3", "all"): experiment3()
    if a.exp in ("3b", "all"): experiment3b()
    if a.exp in ("4", "all"): experiment4()
    if a.exp in ("5", "all"): experiment5()
    if a.exp in ("stats", "all"): run_stats()


if __name__ == "__main__":
    main()
