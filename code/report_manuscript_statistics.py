"""Print the Revision 5 manuscript statistics from the released result files.

This does not refit models or load Qwen. It reads the JSON and CSV files in
results/ that were produced by the analysis scripts.

    python code/report_manuscript_statistics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_paths import RESULTS

ORDER = [
    "H1.1", "H1.2", "H1.3", "H1.4",
    "H2.1", "H2.2", "H2.3",
    "H3.1", "H3.2", "H3.3", "H3.4",
    "H4.2", "H4.3", "H4.4",
]
MEASURES = [
    ("preregistered signed z", "confirmatory_prereg_results.json"),
    ("primary corrected mask fraction", "confirmatory_mask_results.json"),
    ("robustness absolute z", "confirmatory_absz_results.json"),
]
FAMILIES = {
    "H1.1": "F1", "H1.2": "F1", "H1.3": "F1", "H1.4": "F1",
    "H2.1": "F2", "H2.2": "F2", "H2.3": "F2",
    "H3.1": "F3", "H3.2": "F3", "H3.3": "F3", "H3.4": "F3",
    "H4.2": "F4", "H4.3": "F4", "H4.4": "F4",
}


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def fmt_p(p: float) -> str:
    if p >= 0.001:
        return f"{p:.4f}"
    return f"{p:.3e}"


def statistic(block: dict) -> str:
    if "z" in block:
        return f"z = {block['z']:.2f}"
    if "T_sum_delta_dprime" in block:
        return f"sum of delta d' = {block['T_sum_delta_dprime']:.3f}"
    if "W" in block:
        return f"W = {block['W']:.0f}"
    if "rho" in block:
        return f"rho = {block['rho']:.2f}"
    return ""


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    zh = load("baseline_zh/baseline_results.json")
    en = load("baseline_en/baseline_results.json")
    cmp = load("baseline_comparison.json")
    section("Table 2 and Section 4. Corrected baseline")
    print(f"Chinese N = {zh['N']}")
    print("Corrected signature totals, Chinese:", zh["signature_totals"]["fdr"])
    print("  sum =", sum(zh["signature_totals"]["fdr"].values()))
    print("Corrected signature totals, English:", en["signature_totals"]["fdr"])
    print("  sum =", sum(en["signature_totals"]["fdr"].values()))
    zg, eg = zh["circumplex"]["fdr"], en["circumplex"]["fdr"]
    print(f"Graded circumplex rho, Chinese {zg['graded'][0]:.3f} (exact p {zg['graded'][1]:.3f}, {zg['graded'][2]} of 720)")
    print(f"Graded circumplex rho, English  {eg['graded'][0]:.3f} (exact p {eg['graded'][1]:.3f})")
    print(f"Binary circumplex rho, Chinese  {zg['binary'][0]:.3f} (exact p {zg['binary'][1]:.3f})")
    print(f"Partial rho given lexical, Chinese {zg['partial_given_lex'][0]:.3f} (exact p {zg['partial_given_lex'][1]:.3f})")
    print(f"Partial rho given lexical, English  {eg['partial_given_lex'][0]:.3f} (exact p {eg['partial_given_lex'][1]:.3f})")
    print(f"Cophenetic correlation, Chinese {zg['cophenetic']:.3f}, English {eg['cophenetic']:.3f}")
    print(f"Triangle minimum slack, Chinese {zg['triangle_min_slack']:.3f}, English {eg['triangle_min_slack']:.3f}")
    print(f"Top-3 Chinese pairs: {', '.join(zg['top3_pairs'])}")
    cross_r, cross_p = cmp["cross_language_corrected"]
    print(f"Chinese vs English overlap matrices, rho {cross_r:.3f} (exact p {cross_p:.4f})")
    print(
        "Re-extraction vs earlier Jaccard: Spearman "
        f"{cmp['reproduction_original_vs_reextracted_unc_spearman']:.3f}, "
        f"max absolute difference {cmp['reproduction_max_abs_diff_jaccard']:.4f}"
    )
    section("Section 4.5. Sensitivity of the corrected Chinese map")
    for pct, row in zh["sensitivity"].items():
        total = sum(row["signature_total"].values())
        rho, p, _n = row["graded_circumplex"]
        print(f"  threshold {float(pct):4.0f}%  signatures {total:5d}  graded rho {rho:.3f}  exact p {p:.3f}  rho vs 5% {row['rho_vs_5pct']:.3f}")
    section("Table 3. Fourteen preregistered hypotheses")
    for label, filename in MEASURES:
        result = load(filename)
        held = sum(result["confirmatory_holm_p"][h] < 0.05 for h in ORDER)
        print(f"\n{label}: {held} of 14 supported after Holm correction")
        print(f"  {'H':<6} {'statistic':<28} {'raw p':>12} {'Holm p':>12}  held")
        for hyp in ORDER:
            block = result[FAMILIES[hyp]][hyp]
            raw = result["confirmatory_raw_p"][hyp]
            holm = result["confirmatory_holm_p"][hyp]
            decision = "yes" if holm < 0.05 else "no"
            print(f"  {hyp:<6} {statistic(block):<28} {fmt_p(raw):>12} {fmt_p(holm):>12}  {decision}")
    section("Where these values sit in the manuscript")
    print("Signature totals and circumplex rows: Table 2 and Sections 4.2, 4.4, 4.6.")
    print("Cross-language rho: Section 4.9. Triangle slack: Supplementary Section S4.")
    print("Hypothesis rows: Table 3 and Supplementary Table S7.")
    print("The full secondary analyses are inside the same confirmatory JSON files.")


if __name__ == "__main__":
    main()
