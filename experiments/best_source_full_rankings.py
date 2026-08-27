"""Emit the full 24-language E5 rankings as LaTeX table blocks (to stdout and
/tmp/e5_table.tex, /tmp/e5_table_positive.tex, /tmp/e5_table_argmax.tex):
sources ranked by mean score per operationalization, by number of positive
transfer scores, and by number of targets for which the source is the best
choice. Run from the package root."""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import numpy as np
import pandas as pd
from experiments import run_experiments as el
from pipeline import data_sources

s = el.load_setup(sources=("grambank", "wals"))
langs = list(s["langs_ids"])
al = pd.read_csv("data/atlas_languages.csv")
nm = dict(zip(al.glottocode, al.grambank_name))
name = lambda g: nm.get(g, g)

gt = pd.read_csv("data/blt_ground_truth.csv")
emp = gt[gt.glottocode_s.isin(langs) & gt.glottocode_t.isin(langs)].groupby("glottocode_s")["bts"].mean()

deb = pd.read_csv("outputs/debiased__atlas24/debiased_long__atlas24.csv")
br = deb[deb.source.isin(langs) & deb.target.isin(langs)].groupby("source")["bts_T"].mean()

sim = el._grambank.simple_similarity(el._grambank.subset_wide(s["wide"], langs))
cen = pd.Series({g: sim.loc[g, [t for t in langs if t != g]].mean() for g in langs if g in sim.index})

fam = data_sources.wals_family_genus()
def xfamily_targets(g):
    gf = fam.get(g, (None, ""))[0]
    return [t for t in langs if t in fam and fam[t][0] != gf]

def ood(g):
    sub = gt[(gt.glottocode_s == g) & (gt.glottocode_t.isin(xfamily_targets(g)))]
    return sub["bts"].mean() if len(sub) else np.nan
ods = pd.Series({g: ood(g) for g in langs})

cols = [("empirical (raw BTS)", emp),
        ("bias-removed (residual $T$)", br),
        ("centrality (typo.~sim.)", cen),
        ("OOD-generalizer", ods)]
rk = [list(d.reindex(langs).sort_values(ascending=False).index) for _, d in cols]
hdr = " & ".join(n for n, _ in cols)

L = []
L.append(r"\begin{table}[h]")
L.append(r"\centering\small")
L.append(r"\resizebox{\linewidth}{!}{%")
L.append(r"\begin{tabular}{c l r l r l r l r}")
L.append(r"\toprule")
groups = ["empirical", "bias-removed", "centrality", "OOD-generalizer"]
L.append(r"Rank & " + " & ".join(r"\multicolumn{2}{c}{" + g + r"}" for g in groups) + r" \\")
L.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
L.append(r" & source & score & source & score & source & score & source & score \\")
L.append(r"\midrule")
series = [d for _, d in cols]
for i in range(len(langs)):
    cells = []
    for c in range(4):
        g = rk[c][i]
        nm4 = name(g)
        sc = "%.3f" % float(series[c].loc[g])
        cells.append(r"\textbf{" + nm4 + r"}" if i == 0 else nm4)
        cells.append(r"\textbf{" + sc + r"}" if i == 0 else sc)
    L.append("%d & %s \\\\" % (i + 1, " & ".join(cells)))
L.append(r"\bottomrule")
L.append(r"\end{tabular}}")
L.append(r"\caption{Full 24-language ranking of the ATLAS-24 candidate set under each operationalization of Experiment E5.")
L.append(r"The candidate set is the 24 ATLAS-24 languages, used as both sources and targets.")
L.append(r"The columns are the empirical raw-BTS mean, the bias-removed residual $T=\mathrm{BTS}-\hat B$, the mean typological similarity to the targets, and the OOD-generalizer mean BTS to cross-family targets.")
L.append(r"Scores are on each column's own scale and are not comparable across columns; only the within-column order is meaningful.")
L.append(r"The bold top row shows that the four methods disagree on the best source, which is the RQ2 result.}")
L.append(r"\label{tab:e5_fullrank}")
L.append(r"\end{table}")
out = "\n".join(L)
print(out)
with open("/tmp/e5_table.tex", "w") as f:
    f.write(out + "\n")

atlas_pairs = gt[gt.glottocode_s.isin(langs) & gt.glottocode_t.isin(langs)]
emp_n = (atlas_pairs[atlas_pairs.bts > 0].groupby("glottocode_s").size()
         .reindex(langs).fillna(0).astype(int))

atlas_deb = deb[deb.source.isin(langs) & deb.target.isin(langs)]
br_n = (atlas_deb[atlas_deb.bts_T > 0].groupby("source").size()
        .reindex(langs).fillna(0).astype(int))

def ood_pos(g):
    tg = xfamily_targets(g)
    sub = gt[(gt.glottocode_s == g) & (gt.glottocode_t.isin(tg))]
    return int((sub["bts"] > 0).sum()), len(tg)

od_cnt = pd.Series({g: ood_pos(g)[0] for g in langs})
od_den = pd.Series({g: ood_pos(g)[1] for g in langs})

def rank_by_positive(counts, means):
    """Rank langs by positive count desc, then mean desc (tie-break), then
    display name asc (deterministic final tie-break)."""
    return sorted(langs, key=lambda g: (-int(counts[g]), -float(means[g]), name(g)))

pc_rk = [rank_by_positive(emp_n, emp),
         rank_by_positive(br_n, br),
         rank_by_positive(od_cnt, ods)]
pc_counts = [emp_n, br_n, od_cnt]

P = []
P.append(r"\begin{table}[h]")
P.append(r"\centering\small")
P.append(r"\caption{Full 24-language ranking of the ATLAS-24 candidate set by "
         r"positive-transfer coverage: the number of target languages with a "
         r"positive transfer score. The columns are the empirical raw BTS "
         r"(out of 23 targets), the bias-removed residual "
         r"$T=\mathrm{BTS}-\hat B$ (out of 23 targets), and the "
         r"OOD-generalizer raw BTS to cross-family targets (out of that "
         r"source's cross-family target count). Count ties are broken by the "
         r"corresponding mean score in Table~\ref{tab:e5_fullrank}. Counts "
         r"are not comparable across columns; only the within-column order is "
         r"meaningful.}")
P.append(r"\label{tab:e5_posrank}")
P.append(r"\resizebox{\linewidth}{!}{%")
P.append(r"\begin{tabular}{c l r l r l r}")
P.append(r"\toprule")
pgroups = ["empirical", "bias-removed", "OOD-generalizer"]
P.append(r"Rank & " + " & ".join(r"\multicolumn{2}{c}{" + g + r"}" for g in pgroups) + r" \\")
P.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
P.append(r" & source & count & source & count & source & count \\")
P.append(r"\midrule")
for i in range(len(langs)):
    cells = []
    for c in range(3):
        g = pc_rk[c][i]
        nm4 = name(g)
        if c < 2:
            sc = "%d/23" % int(pc_counts[c][g])
        else:
            sc = "%d/%d" % (int(od_cnt[g]), int(od_den[g]))
        cells.append(r"\textbf{" + nm4 + r"}" if i == 0 else nm4)
        cells.append(r"\textbf{" + sc + r"}" if i == 0 else sc)
    P.append("%d & %s \\\\" % (i + 1, " & ".join(cells)))
P.append(r"\bottomrule")
P.append(r"\end{tabular}}")
P.append(r"\end{table}")
outp = "\n".join(P)
print(outp)
with open("/tmp/e5_table_positive.tex", "w") as f:
    f.write(outp + "\n")

def co_best_counts(pairs, scol, src, tgt):
    """{source: number of targets where the source is in the per-target argmax
    set}. Co-best: tied maxima all count, so column sums exceed 24."""
    mx = pairs.groupby(tgt)[scol].transform("max")
    winners = pairs[pairs[scol] == mx]
    return winners.groupby(src).size().reindex(langs).fillna(0).astype(int)

am_emp = co_best_counts(atlas_pairs, "bts", "glottocode_s", "glottocode_t")
am_br = co_best_counts(atlas_deb, "bts_T", "source", "target")

ood_rows = []
for t in langs:
    gft = fam.get(t, (None, ""))[0]
    elig = atlas_pairs[(atlas_pairs.glottocode_t == t)
                       & atlas_pairs.glottocode_s.map(lambda x: x in fam and fam[x][0] != gft)]
    ood_rows.append(elig[elig.bts == elig.bts.max()])
am_ood = (pd.concat(ood_rows).groupby("glottocode_s").size()
          .reindex(langs).fillna(0).astype(int))

# same ordering rule as the positive-coverage table
am_rk = [rank_by_positive(am_emp, emp),
         rank_by_positive(am_br, br),
         rank_by_positive(am_ood, ods)]
am_counts = [am_emp, am_br, am_ood]

A = []
A.append(r"\begin{table}[h]")
A.append(r"\centering\small")
A.append(r"\caption{Full 24-language ranking of the ATLAS-24 candidate set by "
         r"best-source coverage: the number of target languages for which the "
         r"source attains the highest transfer score. The columns are the "
         r"empirical raw BTS, the bias-removed residual "
         r"$T=\mathrm{BTS}-\hat B$, and the raw BTS restricted to cross-family "
         r"sources (the best outside-family option per target). A tie at a "
         r"target's maximum counts for every co-best source, so column counts "
         r"sum to more than 24 (empirically 11 of 24 targets have tied maxima, "
         r"up to 9-way). Count ties between sources are broken by the "
         r"corresponding mean score in Table~\ref{tab:e5_fullrank}. Only the "
         r"within-column order is meaningful.}")
A.append(r"\label{tab:e5_argmax}")
A.append(r"\resizebox{\linewidth}{!}{%")
A.append(r"\begin{tabular}{c l r l r l r}")
A.append(r"\toprule")
agroups = ["empirical", "bias-removed", "OOD-generalizer"]
A.append(r"Rank & " + " & ".join(r"\multicolumn{2}{c}{" + g + r"}" for g in agroups) + r" \\")
A.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
A.append(r" & source & count & source & count & source & count \\")
A.append(r"\midrule")
for i in range(len(langs)):
    cells = []
    for c in range(3):
        g = am_rk[c][i]
        nm4 = name(g)
        sc = "%d/24" % int(am_counts[c][g])
        cells.append(r"\textbf{" + nm4 + r"}" if i == 0 else nm4)
        cells.append(r"\textbf{" + sc + r"}" if i == 0 else sc)
    A.append("%d & %s \\\\" % (i + 1, " & ".join(cells)))
A.append(r"\bottomrule")
A.append(r"\end{tabular}}")
A.append(r"\end{table}")
outa = "\n".join(A)
print(outa)
with open("/tmp/e5_table_argmax.tex", "w") as f:
    f.write(outa + "\n")
