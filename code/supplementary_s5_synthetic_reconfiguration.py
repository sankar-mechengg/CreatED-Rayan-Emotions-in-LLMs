"""Supplementary Section S5 of the Revision 5 manuscript.

Synthetic validation of the five reconfiguration metrics from the earlier
framework: centroid cosine, mean absolute Jaccard change, representational
similarity, mean principal angle, and the change in gating sparsity.

Revision 5 replaced that framework with the four preregistered task families
in run_experiments.py. The paired paradigms in this file, including novice
versus expert explanation, literal versus figurative interpretation, and
cooperative versus adversarial reading, were withdrawn from the experiments.
The synthetic validation is retained because Supplementary Section S5 and
Figure S3 report it. The self-test does not load a language model.

    python supplementary_s5_synthetic_reconfiguration.py --selftest
"""
from __future__ import annotations
import argparse, itertools, json
import numpy as np
from scipy.stats import spearmanr
from scipy.linalg import subspace_angles

# =====================================================================
# (1) PAIRED-CONTEXT STIMULUS SET
# =====================================================================
# Each entry holds semantic content CONSTANT and varies only the objective.
# `probe_items` are the content units; `context_a` / `context_b` are the two
# instructions. Behavioural adaptation is scored by `behaviour_metric`.

PAIRED_CONTEXTS = [
 {
  "paradigm": "role_taking",
  "context_a": "Evaluate the following situation objectively, without adopting "
               "either person's perspective.",
  "context_b": "Take the perspective of the employee described before judging "
               "the following situation.",
  "predicted_change": "Increased weighting of affective and social-state "
                      "representations under B relative to A.",
  "behaviour_metric": "judgement_shift",
  "probe_items": [
    "A manager reassigns an employee's project to a colleague one day before the deadline without explanation.",
    "An employee is passed over for promotion after two years of consistently strong performance reviews.",
    "A worker is asked to stay late for the third consecutive Friday to cover for an absent teammate.",
    "An employee's proposal is presented by their supervisor at a leadership meeting without attribution.",
    "A team member receives critical feedback in an open-plan office within earshot of colleagues.",
    "An employee requesting flexible hours for family reasons is told the team cannot accommodate it.",
  ],
 },
 {
  "paradigm": "hypothesis_generation_vs_discrimination",
  "context_a": "Generate all plausible competing explanations for the following "
               "observation. Do not eliminate any yet.",
  "context_b": "Discriminate between the competing explanations for the "
               "following observation using the evidence given, and commit to a ranking.",
  "predicted_change": "Preserved representational proximity among alternatives "
                      "under A; increased separation along evidence-relevant dimensions under B.",
  "behaviour_metric": "alternative_retention_then_discrimination",
  "probe_items": [
    "A web service shows a 40 percent latency increase that began at 03:00 and affects only European users.",
    "A patient presents with fatigue, low-grade fever, and joint pain persisting for six weeks.",
    "Quarterly revenue fell 12 percent while unit sales rose 4 percent.",
    "A crop yield dropped sharply in one field but not in the adjacent field with the same seed stock.",
    "A machine learning model's accuracy fell after retraining on more data.",
    "Building energy consumption rose 25 percent with no change in occupancy or weather.",
  ],
 },
 {
  "paradigm": "novice_vs_expert",
  "context_a": "Explain the following to a complete novice with no background in the subject.",
  "context_b": "Explain the following to an expert in the field who wants technical precision.",
  "predicted_change": "Collapse of fine technical distinctions into broader "
                      "categories under A; preserved or increased separation under B.",
  "behaviour_metric": "technical_distinction_preservation",
  "probe_items": [
    "The difference between statistical significance and effect size.",
    "How a transformer attention head aggregates information across positions.",
    "Why antibiotic resistance spreads within bacterial populations.",
    "The role of collateral in a repurchase agreement.",
    "How gradient descent finds parameters that minimise a loss function.",
    "The distinction between correlation, confounding, and causation.",
  ],
 },
 {
  "paradigm": "literal_vs_figurative",
  "context_a": "Interpret the following phrase strictly literally.",
  "context_b": "Interpret the following phrase metaphorically, in context.",
  "predicted_change": "Emphasis on concrete semantic relations under A; "
                      "reweighting toward relational and affective associations under B.",
  "behaviour_metric": "interpretation_divergence",
  "probe_items": [
    "The negotiations hit a wall.",
    "Her argument was airtight.",
    "The market is running out of steam.",
    "He carried the whole team on his back.",
    "That policy opened the floodgates.",
    "The evidence points in one direction.",
  ],
 },
 {
  "paradigm": "low_vs_high_stakes",
  "context_a": "Assess the following ambiguity in a low-stakes context where "
               "some uncertainty is acceptable.",
  "context_b": "Assess the same ambiguity in a high-consequence safety-critical "
               "context where low-probability outcomes matter.",
  "predicted_change": "Broader overlap among plausible interpretations under A; "
                      "increased separation of low-probability high-consequence alternatives under B.",
  "behaviour_metric": "risk_sensitivity_shift",
  "probe_items": [
    "A sensor reading is 3 percent outside its expected range.",
    "A batch of components shows slightly inconsistent tolerances.",
    "A software log contains an unexplained intermittent warning.",
    "A structural inspection notes minor surface cracking.",
    "A test result falls just inside the acceptable threshold.",
    "A supplier changed a material specification without notification.",
  ],
 },
 {
  "paradigm": "cooperative_vs_adversarial",
  "context_a": "Read the following statement cooperatively, assuming the speaker "
               "is acting in good faith.",
  "context_b": "Read the following statement adversarially, checking for ways it "
               "could mislead.",
  "predicted_change": "Reweighting toward intent- and trust-related "
                      "representations under B.",
  "behaviour_metric": "charitable_vs_sceptical_reading",
  "probe_items": [
    "Our product reduced costs for the majority of customers who completed the programme.",
    "No significant adverse effects were observed in the trial.",
    "This is the fastest processor in its class.",
    "Most participants reported improvement after the intervention.",
    "We have never had a confirmed security breach.",
    "The results were consistent with our hypothesis.",
  ],
 },
]

EMOTION_CATEGORIES = ["ANGER","DISGUST","FEAR","HAPPINESS","SADNESS","SURPRISE"]


# =====================================================================
# (2) RECONFIGURATION METRICS
# =====================================================================
def signature_sets(A: np.ndarray, top_k: float = 0.05) -> list[set]:
    """A: (n_categories, n_items, n_units) -> per-category signature index sets."""
    k = max(1, int(top_k * A.shape[-1]))
    return [set(np.argsort(np.abs(A[c]).mean(0))[-k:]) for c in range(A.shape[0])]


def jaccard_matrix(A: np.ndarray, top_k: float = 0.05) -> np.ndarray:
    S = signature_sets(A, top_k); n = len(S)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = len(S[i] & S[j]) / max(len(S[i] | S[j]), 1)
    return M


def reconfiguration_metrics(A1: np.ndarray, A2: np.ndarray,
                            top_k: float = 0.05) -> dict:
    """Quantify how relational geometry changes between two task contexts.

    A1, A2 : (n_categories, n_items, n_units) activations under context A and B.
    Returns the five measures recommended in Section 8.3.
    """
    n = A1.shape[0]
    iu = np.triu_indices(n, 1)
    c1, c2 = A1.mean(1), A2.mean(1)

    centroid_cos = float(np.mean([
        np.dot(c1[i], c2[i]) / (np.linalg.norm(c1[i]) * np.linalg.norm(c2[i]))
        for i in range(n)]))

    J1, J2 = jaccard_matrix(A1, top_k), jaccard_matrix(A2, top_k)
    jaccard_delta = float(np.abs(J1[iu] - J2[iu]).mean())
    rsa_rho = float(spearmanr(J1[iu], J2[iu]).statistic)
    subspace_deg = float(np.rad2deg(subspace_angles(c1.T, c2.T)).mean())

    q1 = np.quantile(np.abs(A1), 1 - top_k)
    q2 = np.quantile(np.abs(A2), 1 - top_k)
    sparsity_delta = float(abs((np.abs(A1) > q1).mean() - (np.abs(A2) > q2).mean()))

    return dict(centroid_cosine=centroid_cos,
                jaccard_delta=jaccard_delta,
                rsa_rho=rsa_rho,
                subspace_angle_deg=subspace_deg,
                sparsity_delta=sparsity_delta,
                jaccard_A=J1.tolist(), jaccard_B=J2.tolist())


def null_threshold(extract_fn, texts_by_cat, context, n_boot=30,
                   top_k=0.05, alpha=0.05, seed=0) -> float:
    """Split-half null: RSA between two independent halves of the SAME context.

    Any observed cross-context RSA below this threshold indicates
    reconfiguration exceeding item-sampling noise.
    """
    rng = np.random.default_rng(seed)
    n = len(texts_by_cat)
    iu = np.triu_indices(n, 1)
    vals = []
    for _ in range(n_boot):
        h1, h2 = [], []
        for c in range(n):
            idx = rng.permutation(len(texts_by_cat[c]))
            half = len(idx) // 2
            h1.append(extract_fn([texts_by_cat[c][i] for i in idx[:half]], context))
            h2.append(extract_fn([texts_by_cat[c][i] for i in idx[half:2*half]], context))
        J1 = jaccard_matrix(np.stack(h1), top_k)
        J2 = jaccard_matrix(np.stack(h2), top_k)
        vals.append(spearmanr(J1[iu], J2[iu]).statistic)
    return float(np.quantile(vals, alpha))


def behavioural_adaptation(responses_a: list[str], responses_b: list[str],
                           scorer) -> float:
    """Magnitude of behavioural change between contexts. `scorer` is a
    task-specific callable returning a scalar per response (Section 8.4)."""
    sa = np.array([scorer(r) for r in responses_a], dtype=float)
    sb = np.array([scorer(r) for r in responses_b], dtype=float)
    pooled = np.sqrt((sa.var(ddof=1) + sb.var(ddof=1)) / 2)
    return float((sb.mean() - sa.mean()) / pooled) if pooled > 0 else 0.0


def run_protocol(extract_fn, paradigms=None, top_k=0.05) -> dict:
    """Full pipeline. Returns per-paradigm reconfiguration metrics.

    The central test of Section 8.4 is whether the magnitude of
    representational reconfiguration predicts the magnitude of behavioural
    adaptation ACROSS paradigms.
    """
    out = {}
    for spec in (paradigms or PAIRED_CONTEXTS):
        items = spec["probe_items"]
        A = np.stack([extract_fn(items, spec["context_a"])])
        B = np.stack([extract_fn(items, spec["context_b"])])
        out[spec["paradigm"]] = reconfiguration_metrics(A, B, top_k)
    return out


def hf_extractor(model_name="Qwen/Qwen2.5-1.5B-Instruct", layers=range(14, 23)):
    """Reference activation extractor for HuggingFace SwiGLU models.

    NOT executed in the supplementary self-test: requires GPU hardware.
    Captures the post-SwiGLU gating product swish(gate_proj(x)) * up_proj(x),
    mean-pooled over non-padding tokens, concatenated across `layers`.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto").eval()
    store = {}

    def hook(idx):
        def f(mod, inp, out):
            store[idx] = (mod.act_fn(mod.gate_proj(inp[0])) * mod.up_proj(inp[0])).detach()
        return f
    for i in layers:
        model.model.layers[i].mlp.register_forward_hook(hook(i))

    def extract(texts, context_prompt):
        vecs = []
        for txt in texts:
            prompt = f"{context_prompt}\n\n{txt}"
            enc = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**enc)
            v = torch.cat([store[i][0].float().mean(0) for i in layers])
            vecs.append(v.cpu().numpy())
        return np.stack(vecs)
    return extract


# =====================================================================
# (4) SYNTHETIC SELF-TEST WITH KNOWN GROUND TRUTH
# =====================================================================
def selftest(n_units=512, n_cat=6, top_k=0.05, n_sim=30, verbose=True):
    """Validate that the metrics recover a KNOWN imposed reconfiguration and
    quantify detection power. Reproduces Figure S3 of the Revision 5 supplement."""
    G = np.random.default_rng(7)
    back = np.abs(G.normal(0, 1, n_units))
    v1, v2 = G.normal(0, 1, n_units), G.normal(0, 1, n_units)
    idio = G.normal(0, 1, (n_cat, n_units))
    shift = np.array([0.0, 0.9, 0.0, 0.9, 0.55, 0.55])
    iu = np.triu_indices(n_cat, 1)

    def centroids(delta):
        th = 2 * np.pi * np.arange(n_cat) / n_cat + delta * shift
        return (0.9 * back[None, :]
                + 1.6 * (np.cos(th)[:, None] * v1[None, :] + np.sin(th)[:, None] * v2[None, :])
                + 0.5 * idio)

    def sample(delta, n_items, seed):
        g = np.random.default_rng(seed); C = centroids(delta)
        return np.stack([C[c] + g.normal(0, 1.0, (n_items, n_units)) for c in range(n_cat)])

    results = {"responsiveness": [], "power": {}}
    if verbose:
        print(f"{'delta':>7} | {'centroid cos':>12} | {'|dJaccard|':>11} | {'RSA rho':>8}")
    for delta in [0.0, 0.1, 0.25, 0.5, 1.0]:
        ms = [reconfiguration_metrics(sample(0.0, 120, 300 + s),
                                      sample(delta, 120, 800 + s), top_k)
              for s in range(8)]
        agg = {k: float(np.mean([m[k] for m in ms]))
               for k in ("centroid_cosine", "jaccard_delta", "rsa_rho", "subspace_angle_deg")}
        results["responsiveness"].append((delta, agg))
        if verbose:
            print(f"{delta:>7.2f} | {agg['centroid_cosine']:>12.4f} | "
                  f"{agg['jaccard_delta']:>11.4f} | {agg['rsa_rho']:>8.4f}")

    if verbose:
        print("\nDetection power (null = two draws of the SAME context, alpha=0.05):")
    for n_items in [20, 50, 100, 200]:
        null = [spearmanr(jaccard_matrix(sample(0, n_items, 10_000 + s), top_k)[iu],
                          jaccard_matrix(sample(0, n_items, 50_000 + s), top_k)[iu]).statistic
                for s in range(n_sim)]
        thr = float(np.quantile(null, 0.05)); line = []
        for delta in [0.1, 0.25, 0.5]:
            det = sum(1 for s in range(n_sim)
                      if spearmanr(jaccard_matrix(sample(0, n_items, 20_000 + s), top_k)[iu],
                                   jaccard_matrix(sample(delta, n_items, 60_000 + s), top_k)[iu]
                                   ).statistic < thr)
            results["power"][f"{n_items}_{delta}"] = det / n_sim
            line.append(f"d={delta}: {det / n_sim * 100:>3.0f}%")
        if verbose:
            print(f"  n={n_items:>4} (null 5th pct={thr:.3f}): " + "  ".join(line))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-stimuli", action="store_true")
    a = ap.parse_args()
    if a.dump_stimuli:
        print(json.dumps(PAIRED_CONTEXTS, indent=2))
    elif a.selftest:
        selftest()
    else:
        n = sum(len(s["probe_items"]) for s in PAIRED_CONTEXTS)
        print(f"{len(PAIRED_CONTEXTS)} paired paradigms, {n} probe items.")
        print("Run with --selftest to validate metrics, --dump-stimuli to export.")
