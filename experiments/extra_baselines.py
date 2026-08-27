# extra_baselines.py
"""Five extra distance baselines for the ATLAS-24 BTS matrix: lang2vec featural
cosine, the six URIEL distances, and three Wikipedia-corpus lexical overlaps
(subword tokenizer, character n-gram, word vocabulary). Pair tables are built
once (`prepare`) and committed; scoring (`run`) uses the same LOLO protocol
as fit_models._run_baselines and writes data/d2_model_comparison__extra.csv.
Nothing in fit_models.py or the existing committed tables is touched."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parents[1]
DATA = HERE / "data"
CORPUS_DIR = DATA / "wikipedia_text"
TOKENIZER_DIR = DATA / "tokenizers"

VOCAB_TOP = 10_000     # word types per language for vocab_overlap
NGRAM_TOP = 5_000      # word types feeding char_ngrams
NGRAM_SIZES = (1, 2, 3)
BPE_VOCAB = 8_000      # byte-level BPE vocabulary entries per language

L2V_SETS = ("syntax_knn", "phonology_knn", "inventory_knn")
URIEL_SETS = {"genetic": "fam", "syntactic": "syntax_knn",
              "phonological": "phonology_knn", "inventory": "inventory_knn"}
GEO_MAX_KM = 20015.09  # half great-circle circumference; normalizes geo to [0, 1]


def iso_codes():
    """{atlas_name: ISO 639-3} from the committed data/iso6393.csv."""
    with open(DATA / "iso6393.csv", newline="") as f:
        return {r["atlas_name"]: r["iso6393"] for r in csv.DictReader(f)}


def count_words(text):
    """Counter over word types: NFKC-normalized, casefolded, digit-only
    tokens dropped (numbers are script-neutral noise for vocabulary overlap)."""
    norm = unicodedata.normalize("NFKC", text).casefold()
    return Counter(t for t in re.findall(r"\w+", norm) if not t.isdigit())


def top_types(counts, n):
    """The n most frequent types (Counter.most_common; ties keep order).
    Accepts any mapping or iterable of types; Counter() over a Counter copies."""
    return {w for w, _ in Counter(counts).most_common(n)}


def char_ngrams(words, ns=NGRAM_SIZES):
    """Set of character n-grams over space-padded words, n in ns."""
    out = set()
    for w in words:
        padded = " %s " % w
        for n in ns:
            if len(padded) >= n:
                out.update(padded[i:i + n] for i in range(len(padded) - n + 1))
    return out


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def train_bpe(text, vocab_size=BPE_VOCAB):
    """Byte-level BPE vocabulary (set of tokens) trained on one language's
    text. Byte-level (GPT-2 recipe) so any script encodes without an unk
    token; deterministic given the same text and vocab_size."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False)
    tok.train_from_iterator([text], trainer)
    return set(tok.get_vocab())


def fetch_l2v(iso_langs):
    """{set_name: {iso: ndarray}} for every set the baselines need. Downloads
    lang2vec data on first use (cached in the package install dir)."""
    import lang2vec.lang2vec as l2v
    sets = set(L2V_SETS) | set(URIEL_SETS.values())
    out = {}
    for name in sorted(sets):
        feats = l2v.get_features(list(iso_langs), name)
        out[name] = {k: np.asarray(v, dtype=float) for k, v in feats.items()}
    return out


def _cosine(a, b):
    """NaN-safe cosine; NaN when either vector has no usable coordinates."""
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any() or np.linalg.norm(a[m]) == 0 or np.linalg.norm(b[m]) == 0:
        return float("nan")
    return float(np.dot(a[m], b[m]) / (np.linalg.norm(a[m]) * np.linalg.norm(b[m])))


def geo_distance_table(lang_rows, data=None):
    """{(s, t): normalized great-circle distance} for the URIEL geographic
    distance. lang2vec's 'geo' set is a 299-dim distance-to-reference-point
    vector, not per-language coordinates, so coordinates come from Glottolog
    via the Grambank CLDF (the same source the glm tier's geographic distance
    uses), normalized by half the great-circle circumference. `data` injects
    the (values, params, langs, codes) CLDF tuple for tests; when None it
    loads the CLDF."""
    from pipeline import _grambank
    if data is None:
        data = _grambank.load_data()
    langs = data[2]
    km = _grambank.geographic_distance(langs, [g for g, _, _ in lang_rows])
    return {(s, t): float(km.loc[s, t]) / GEO_MAX_KM
            for s, _, _ in lang_rows for t, _, _ in lang_rows if s != t}


def build_l2v_tables(lang_rows, vectors=None, geo=None):
    """(l2v_similarity_df, uriel_distances_df) over all ordered glottocode
    pairs. lang_rows: (glottocode, wiki_code, iso) triples. `vectors` injects
    {set: {iso: ndarray}} and `geo` injects {(s, t): normalized distance} for
    tests; when None they are fetched/computed live."""
    if vectors is None:
        vectors = fetch_l2v(sorted({iso for _, _, iso in lang_rows}))
    if geo is None:
        geo = geo_distance_table(lang_rows)
    sim_rows, uriel_rows = [], []
    for s_row in lang_rows:
        for t_row in lang_rows:
            s, _, i_s = s_row
            t, _, i_t = t_row
            if s == t:
                continue
            d = {name: 1.0 - _cosine(vectors[setname][i_s], vectors[setname][i_t])
                 for name, setname in URIEL_SETS.items()}
            d["geographic"] = float(geo[(s, t)])
            d["featural"] = float(np.mean([d["syntactic"], d["phonological"],
                                           d["inventory"]]))
            sim_rows.append({
                "glottocode_s": s, "glottocode_t": t,
                "lang2vec_cosine": _cosine(
                    np.concatenate([vectors[k][i_s] for k in L2V_SETS]),
                    np.concatenate([vectors[k][i_t] for k in L2V_SETS]))})
            uriel_rows.append({"glottocode_s": s, "glottocode_t": t, **d})
    return pd.DataFrame(sim_rows), pd.DataFrame(uriel_rows)


def build_lexical_table(lang_rows, corpus_dir=CORPUS_DIR):
    """Corpus-derived pair table (vocab_overlap, ngram_overlap, tok_overlap)
    over all ordered glottocode pairs. Reads <wiki_code>.txt.gz per language."""
    per = {}
    for g, code, _ in lang_rows:
        with gzip.open(Path(corpus_dir) / ("%s.txt.gz" % code), "rt",
                       encoding="utf-8") as f:
            text = f.read()
        counts = count_words(text)
        per[g] = {"vocab": top_types(counts, VOCAB_TOP),
                  "ngrams": char_ngrams(top_types(counts, NGRAM_TOP)),
                  "bpe": train_bpe(text)}
    rows = []
    for s, _, _ in lang_rows:
        for t, _, _ in lang_rows:
            if s == t:
                continue
            rows.append({"glottocode_s": s, "glottocode_t": t,
                         "vocab_overlap": jaccard(per[s]["vocab"], per[t]["vocab"]),
                         "ngram_overlap": jaccard(per[s]["ngrams"], per[t]["ngrams"]),
                         "tok_overlap": jaccard(per[s]["bpe"], per[t]["bpe"])})
    return pd.DataFrame(rows)


def modeling_pairs(floor=0.70):
    """((glottocode, wiki_code, iso) triples, {(s, t): bts}) for the canonical
    ATLAS modeling set. Requires the Grambank CLDF (floor filtering), like
    every other entry point."""
    from pipeline import atlas_languages
    from data_setup.download_wikipedia import wiki_code
    iso = iso_codes()
    keep = set(atlas_languages.atlas_glottocodes(floor))
    rows = []
    for r in atlas_languages.load_mapping().to_dict("records"):
        name = r.get("atlas_name")
        name = name.strip() if isinstance(name, str) else ""
        g = r.get("glottocode")
        g = g.strip() if isinstance(g, str) else ""
        if name in iso and g in keep:
            rows.append((g, wiki_code(name), iso[name]))
    bts = pd.read_csv(DATA / "blt_ground_truth.csv").dropna(
        subset=["bts", "glottocode_s", "glottocode_t"])
    bts = bts[bts["glottocode_s"].isin(keep) & bts["glottocode_t"].isin(keep)]
    lookup = {(r.glottocode_s, r.glottocode_t): float(r.bts)
              for r in bts.itertuples()}
    return rows, lookup


URIEL_COLS = ["genetic", "geographic", "syntactic", "phonological",
              "inventory", "featural"]


def run_extra_baselines(lang_rows, lookup, l2v_sim, uriel, lexical, out_csv):
    """Score the five tiers under the protocol of fit_models._run_baselines:
    directed pairs per tier's covered universe, glottocode-sorted like
    fit_models._load_common so cv_predict's last-write-wins fold matches the
    existing tables, leave-one-language-out folds,
    within-fold StandardScaler+Ridge(alpha=1.0); LOLO Pearson + true R^2 per
    tier. Writes out_csv (model, n, lolo_pearson, lolo_r2) and returns it."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from pipeline.fit_models import cv_predict, directed_pairs, lolo_folds, pearson, r2
    tiers = [("lang2vec", l2v_sim, ["lang2vec_cosine"]),
             ("uriel", uriel, URIEL_COLS),
             ("tok_overlap", lexical, ["tok_overlap"]),
             ("ngram_overlap", lexical, ["ngram_overlap"]),
             ("vocab_overlap", lexical, ["vocab_overlap"])]
    results = []
    for name, table, cols in tiers:
        have = set(table["glottocode_s"]) | set(table["glottocode_t"])
        langs = sorted(g for g, _, _ in lang_rows if g in have)
        pairs = directed_pairs(langs)
        y = np.array([lookup[(s, t)] for s, t in pairs])
        idx = table.set_index(["glottocode_s", "glottocode_t"])
        X = np.vstack([idx.loc[(s, t), cols].to_numpy(dtype=float)
                       for s, t in pairs])
        pred = cv_predict(X, y, lolo_folds(pairs, langs),
                          lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)))
        results.append({"model": name, "n": len(langs),
                        "lolo_pearson": pearson(y, pred), "lolo_r2": r2(y, pred)})
        print("  %s: n=%d rho=%.4f R2=%.4f" % (
            name, len(langs), results[-1]["lolo_pearson"], results[-1]["lolo_r2"]))
    out = pd.DataFrame(results)
    out.to_csv(out_csv, index=False)
    return out


def write_provenance(path, manifest_path=DATA / "wikipedia_text_manifest.csv"):
    """Parameter + provenance snapshot: versions and corpus-manifest hash."""
    import tokenizers
    prov = {"params": {"vocab_top": VOCAB_TOP, "ngram_top": NGRAM_TOP,
                       "ngram_sizes": list(NGRAM_SIZES), "bpe_vocab": BPE_VOCAB},
            "tokenizers_version": tokenizers.__version__,
            "corpus_manifest_sha256":
                hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
            if Path(manifest_path).exists() else None}
    try:
        import lang2vec
        prov["lang2vec_version"] = getattr(lang2vec, "__version__", "unknown")
    except ImportError:
        prov["lang2vec_version"] = "not-installed"
    Path(path).write_text(json.dumps(prov, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["prepare", "run"],
                    help="prepare: build+commit pair tables; run: LOLO scoring")
    args = ap.parse_args(argv)
    lang_rows, lookup = modeling_pairs()
    if args.step == "prepare":
        l2v_sim, uriel = build_l2v_tables(lang_rows)
        # NaN in a committed pair table would silently poison the ridge fit;
        # _cosine returns NaN on zero-norm vectors and blank CLDF coordinates
        # produce NaN geo entries, so fail loudly before writing anything.
        for name, table in (("l2v_similarity", l2v_sim), ("uriel_distances", uriel)):
            if not np.isfinite(table.drop(columns=["glottocode_s", "glottocode_t"])
                               .to_numpy(dtype=float)).all():
                raise SystemExit("%s contains non-finite values" % name)
        l2v_sim.to_csv(DATA / "l2v_similarity.csv", index=False)
        uriel.to_csv(DATA / "uriel_distances.csv", index=False)
        build_lexical_table(lang_rows).to_csv(DATA / "lexical_overlap.csv",
                                              index=False)
        write_provenance(DATA / "extra_baselines_provenance.json")
        print("wrote l2v_similarity.csv, uriel_distances.csv, lexical_overlap.csv")
    else:
        missing = [f for f in ("l2v_similarity.csv", "uriel_distances.csv",
                               "lexical_overlap.csv")
                   if not (DATA / f).exists()]
        if missing:
            raise SystemExit("missing %s - run `uv run python experiments/extra_baselines.py "
                             "prepare` first" % ", ".join(missing))
        tables = [pd.read_csv(DATA / f) for f in
                  ("l2v_similarity.csv", "uriel_distances.csv",
                   "lexical_overlap.csv")]
        run_extra_baselines(lang_rows, lookup, *tables,
                            DATA / "d2_model_comparison__extra.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
