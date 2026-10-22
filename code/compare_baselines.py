"""
Builds (i) the supplementary comparison of original (R3) values, the re-extracted
uncorrected values and the corrected (BH-FDR + Cohen's h) values, and (ii) the
Chinese versus English cross-language replication of the corrected overlap
structure (Spearman over the 15 pairs, exact permutation over 720 relabellings;
per-emotion overlap of signature sets between languages).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_paths import (
    DATA_CSV, EARLIER_OVERLAP, CACHE, RESULTS, STIMULI, REFERENCE, FIGURES,
    PIPELINE, load_nrc_lexicon, text_open, apply_figure_style, ensure_runtime_dirs,
)
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(STIMULI))
ensure_runtime_dirs()

import json, itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
EM = ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE"]
LAYERS = list(range(14, 23))
iu = np.triu_indices(6, 1); PERMS = list(itertools.permutations(range(6)))
R = str(RESULTS)
zh = json.load(text_open(f"{R}/baseline_zh/baseline_results.json"))
en = json.load(text_open(f"{R}/baseline_en/baseline_results.json"))
orig_J = pd.read_csv(str(EARLIER_OVERLAP), index_col=0).loc[EM, EM].values
orig_tot = {"ANGER": 14165, "DISGUST": 9958, "FEAR": 13121, "HAPPINESS": 21555, "SADNESS": 14829, "SURPRISE": 23115}
orig_excl = {"ANGER": 5917, "DISGUST": 2449, "FEAR": 3759, "HAPPINESS": 10050, "SADNESS": 4685, "SURPRISE": 13012}

def exact(a, b):
    r = spearmanr(a[iu], b[iu]).statistic
    null = np.array([spearmanr(a[np.ix_(p, p)][iu], b[iu]).statistic for p in PERMS])
    return float(r), float(np.mean(null >= r - 1e-12))

out = {}
Jz = {k: np.array(zh["jaccard"][k]) for k in ("fdr", "unc")}
Je = {k: np.array(en["jaccard"][k]) for k in ("fdr", "unc")}
rows = []
for i, j in zip(*iu):
    rows.append({"pair": f"{EM[i]}-{EM[j]}", "original_R3": orig_J[i, j], "reextracted_uncorrected_zh": Jz["unc"][i, j],
                 "corrected_zh": Jz["fdr"][i, j], "corrected_en": Je["fdr"][i, j]})
tabJ = pd.DataFrame(rows).sort_values("original_R3", ascending=False)
tabJ.to_csv(f"{R}/supp_table_jaccard_original_vs_corrected.csv", index=False)
rows = []
for e in EM:
    rows.append({"emotion": e, "original_total": orig_tot[e], "reextracted_uncorrected_total": zh["signature_totals"]["unc"][e],
                 "corrected_total_zh": zh["signature_totals"]["fdr"][e], "corrected_total_en": en["signature_totals"]["fdr"][e],
                 "original_exclusive": orig_excl[e], "corrected_exclusive_zh": zh["exclusive_shared"]["fdr"][e]["exclusive"],
                 "corrected_shared_zh": zh["exclusive_shared"]["fdr"][e]["shared"]})
tabS = pd.DataFrame(rows); tabS.to_csv(f"{R}/supp_table_signature_counts.csv", index=False)
out["reproduction_original_vs_reextracted_unc_spearman"] = float(spearmanr(orig_J[iu], Jz["unc"][iu]).statistic)
out["reproduction_max_abs_diff_jaccard"] = float(np.abs(orig_J[iu] - Jz["unc"][iu]).max())
out["corrected_vs_original_zh"] = exact(Jz["fdr"], orig_J)
out["cross_language_corrected"] = exact(Je["fdr"], Jz["fdr"])
out["cross_language_uncorrected"] = exact(Je["unc"], Jz["unc"])
sz = np.load(f"{R}/baseline_zh/signatures.npz"); se = np.load(f"{R}/baseline_en/signatures.npz")
per = {}
for e in EM:
    a = set(); b = set()
    for l in LAYERS:
        k = f"fdr_{l}_{e}"
        if k in sz.files:
            a |= {(l, int(x)) for x in np.flatnonzero(sz[k])}; b |= {(l, int(x)) for x in np.flatnonzero(se[k])}
    per[e] = {"zh": len(a), "en": len(b), "shared": len(a & b), "jaccard": len(a & b) / max(1, len(a | b))}
out["per_emotion_signature_overlap_zh_en"] = per
out["en_circumplex"] = en["circumplex"]["fdr"]; out["zh_circumplex"] = zh["circumplex"]["fdr"]
json.dump(out, text_open(f"{R}/baseline_comparison.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3000]); print(tabJ.round(4).to_string()); print(tabS.to_string())
