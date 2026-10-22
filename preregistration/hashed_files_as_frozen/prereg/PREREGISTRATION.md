# Preregistration: context-dependent reconfiguration of emotion-selective FFN units in Qwen2.5-1.5B-Instruct

Status: frozen before any experimental activation was extracted. The SHA-256 hash of this file, of the stimulus files, of the experiment code and of the analysis code, together with a UTC timestamp, is recorded in `prereg/HASHES.txt`. The hash file was written before `run_experiments.py F1` was started. Any change after that point is listed under "Deviations" in the manuscript supplement.

Registration type: internal, time-stamped and hashed (not uploaded to a public registry). This is disclosed in the manuscript.

## 1. Background and purpose

The corrected baseline (Benjamini-Hochberg FDR, q < 0.05, within each layer and emotion family of 8,960 tests, plus Cohen's h of at least 0.2) identifies emotion-selective neurons in layers 14 to 22 of Qwen2.5-1.5B-Instruct and an overlap structure among the six emotion categories. The question tested here is whether the same signature map tracks task-dependent changes in the model's internal emotion representation, and whether those internal changes relate to the model's behaviour, in four task families run on stimuli that were not used to derive the map.

## 2. Model and settings

* Model: Qwen/Qwen2.5-1.5B-Instruct, bfloat16 weights, CPU, PyTorch 2.14.1, transformers 5.18.0.
* Prompts use the model's chat template with one user turn. The task instruction is placed before the stimulus, so that the stimulus tokens are processed in the context of the instruction.
* Forced-choice tasks: the assistant turn is prefixed with `Answer:` and the next-token log-probabilities of the six single-token label words (" anger", " disgust", " fear", " happiness", " sadness", " surprise") are read out.
* Generation tasks: greedy decoding, at most 80 new tokens, seed 42.

## 3. Reference map (fixed before the experiments)

* English translations of the 15,895 deduplicated texts, stratified 70/30 split into a derivation set and a held-out set (scikit-learn `train_test_split`, `random_state=42`).
* Signatures: per layer and emotion, top-5% absolute activation masks, one-sided two-proportion z-test, BH-FDR q < 0.05 within each (layer, emotion) family, Cohen's h of at least 0.2, target rate above other rate. Derivation set only.
* z-scale: per-neuron mean and standard deviation of mean-pooled activations in the derivation set.
* Emotion score of a token: s_e = sum over layers of z(layer) dot W_e(layer), where W_e places weight 1/|S_e| on every signature neuron of emotion e (S_e aggregated over the nine layers). The pair axis for a pair (a, b) is A_ab = s_a minus s_b.
* Reference overlap: Jaccard index between the aggregated (layer, neuron) signature sets of the six emotions, derivation split. Values are stored in `results/reference_map_jaccard_en.json` and reproduced in Section 9.

## 4. Stimuli

* Cross-language check of the baseline, run before freezing: the corrected Chinese and English overlap matrices correlate at Spearman 0.92 (exact p = 1/720); this is reported in the manuscript and does not alter any hypothesis.
* Family 1: 80 held-out texts per emotion for each member of the three high-overlap pairs (anger and disgust, fear and sadness, happiness and surprise), 6 to 60 English words, sampled with seed 42 (480 items; 160 per pair). Gold label: the human primary label.
* Family 2: 40 newly written items per pair (120 items). Each item has an ambiguous base account, one evidence sentence favouring each member of the pair, and a neutral filler sentence. Evidence sentences contain no emotion label word or direct morphological variant (checked by regular expression; 0 violations).
* Family 3: 40 newly written neutral factual statements, each rewritten towards each of the six emotions (240 item by target cells).
* Family 4: 40 situations that call for anger and 40 that call for fear (80 items), each written up under three conditions: EXPRESS, SUPPRESS, NEUTRAL.
* Every instruction has four paraphrases. In forced-choice tasks the order of the two options is reversed on odd-numbered paraphrases.
* Family 2 to 4 stimuli were written by an AI system (Claude, Anthropic) under the direction of the authors; this is disclosed.

## 5. Design and measures

### Family 1: recognition versus forced pair discrimination

Conditions: RECOGNISE (open instruction to identify any emotions) and DISCRIMINATE (choose between the two members of the item's pair). 480 items x 2 conditions x 4 paraphrases = 3,840 trials.

* Internal measure: pair axis averaged over the stimulus tokens, signed towards the gold label (`signed`).
* Behaviour: log-probability of the gold label word minus that of the other pair member; correct if positive.

### Family 2: disambiguation after new evidence

Conditions: NONE (base only), FILLER (base plus neutral sentence), EV_A (base plus evidence for the first member), EV_B (base plus evidence for the second member). 120 items x 4 conditions x 4 paraphrases = 1,920 trials.

* Internal measure: pair axis A_ab at the final prompt token (after `Answer:`).
* Behaviour: log-probability of label a minus that of label b.

### Family 3: targeted tone generation

240 cells x 4 paraphrases = 960 generations.

* Internal measure: emotion scores averaged over the generated tokens, re-read in a forward pass over prompt plus output, centred within item and paraphrase across the six targets. Target margin = centred target score minus mean centred score of the other five.
* Behaviour: j-hartmann/emotion-english-distilroberta-base probabilities (joy mapped to happiness; neutral excluded from the six-way renormalisation). Second scorer: NRC Emotion Lexicon as bundled in nrclex 4.1.0 (word proportions, no lemmatisation).

### Family 4: expression versus suppression of anger and fear

80 items x 3 conditions x 4 paraphrases = 960 generations.

* Internal measures: target emotion score over the generated tokens (`gen_t`), and at the final prompt token before generation (`plan_t`, planning state).
* Behaviour: classifier probability of the target emotion (`Pt`).

## 6. Confirmatory hypotheses (all one-sided)

Mixed models use crossed random intercepts for item and paraphrase (statsmodels MixedLM, REML; binary outcomes with BinomialBayesMixedGLM, variational Bayes, where the "p value" is the normal tail of posterior mean over posterior SD).

* H1.1: `signed` is higher under DISCRIMINATE than RECOGNISE. Model: signed ~ D + pair + (1|item) + (1|paraphrase). Test: coefficient of D > 0.
* H1.2: the paraphrase-level d' of the pair axis between the two gold classes is higher under DISCRIMINATE than RECOGNISE, summed over the three pairs. Null: exact permutation of the condition labels over the eight paraphrase-condition cells within each pair (70 per pair, 343,000 combinations).
* H1.3: in DISCRIMINATE trials, `signed` (z-scored) predicts behavioural correctness. Model: correct ~ zsig + pair + (1|item) + (1|paraphrase). Coefficient > 0.
* H1.4: the magnitude of reconfiguration predicts the behavioural gain. Unit: item by paraphrase index (DISCRIMINATE paraphrase k paired with RECOGNISE paraphrase k). Change in `signed` (DISCRIMINATE minus RECOGNISE, z-scored) predicts change in the behavioural margin. Model: dB ~ zdS + pair + (1|item) + (1|paraphrase). Coefficient > 0.
* H2.1: A_ab rises with evidence direction (EV_A = +1, EV_B = minus 1, NONE and FILLER = 0). Model: A ~ evdir + pair + (1|item) + (1|paraphrase). Coefficient > 0.
* H2.2: per item (averaged over paraphrases), half the evidence-driven shift (EV_A minus EV_B, halved) exceeds the absolute shift produced by the neutral filler (FILLER minus NONE). Wilcoxon signed-rank test, greater.
* H2.3: across item by paraphrase units, the internal shift (EV_A minus EV_B, z-scored) predicts the behavioural shift. Model: Bshift ~ zI + pair + (1|item) + (1|paraphrase). Coefficient > 0.
* H3.1: the centred target margin over generated tokens is positive. Model: margin ~ 1 + (1|item) + (1|paraphrase). Intercept > 0.
* H3.2: across the 15 emotion pairs, internal separation of generations (mean target minus competitor score for both directions) is negatively rank-correlated with reference overlap (Jaccard). Exact permutation over the 720 relabellings of the six emotions.
* H3.3: as H3.2 for behavioural separation (classifier).
* H3.4: the target margin (z-scored) predicts classifier correctness (top label equals target). Model: correct ~ zm + (1|item) + (1|paraphrase). Coefficient > 0.
* H4.2: `gen_t` is higher under EXPRESS than SUPPRESS. Model: gen_t ~ E + target + (1|item) + (1|paraphrase). Coefficient > 0.
* H4.3: the EXPRESS minus SUPPRESS difference is larger over generated tokens than at the planning state. Long format with z-scored measures; model: y ~ E x stage + target + (1|item) + (1|paraphrase) + (1|trial). Interaction > 0.
* H4.4: `gen_t` (z-scored) predicts `Pt` across all three conditions. Model: Pt ~ zg + condition + target + (1|item) + (1|paraphrase). Coefficient > 0.

Multiplicity: Holm correction across these 14 tests at family-wise alpha 0.05. H4.1 (classifier `Pt` higher under EXPRESS than SUPPRESS) is a manipulation check and is reported outside the Holm family.

Interpretation rule: a hypothesis is supported if its Holm-adjusted p is below 0.05. A non-significant result is reported as a failure to support the hypothesis, not as evidence of absence. Results are reported whatever their direction, and the title and abstract are written after the results.

## 7. Secondary and exploratory analyses (not in the Holm family)

* Family 1: mean D minus R change for each of the four paraphrase indices, and whether all four are positive; final-token variant of H1.1; per-pair H1.1; H1.1 in odd and even item halves; item-level Spearman between the change in internal separation and the change in behaviour; accuracy by pair and condition.
* Family 2: mean evidence-driven shift for each paraphrase; per-pair H2.1; H2.1 in item halves; proportion of trials in which behaviour followed the evidence; proportion of item by paraphrase units with positive internal shift.
* Family 3: classifier accuracy overall and by target; correlation between internal and behavioural separation matrices; planning-state margin; NRC lexicon separation against overlap; NRC hit rate.
* Family 4: EXPRESS minus SUPPRESS on `gen_t` for each paraphrase; planning-state EXPRESS minus SUPPRESS; SUPPRESS versus NEUTRAL for `gen_t` and `plan_t`; NRC manipulation check; condition means; neighbour shift (disgust minus anger score for anger items and sadness minus fear for fear items) by condition.

## 8. Exclusions and data handling

* Generation trials with empty output are excluded and counted. No other trial or item exclusions, no outlier removal.
* All trials, outputs, scores and pooled raw activations are saved. If a run is interrupted it resumes from the saved trial list; completed trials are not recomputed.
* A pilot was run before freezing, only to check the pipeline and time generation. The pilot used no study stimuli (a sentence about a rainy afternoon and a prompt about a bicycle bell). Pilot timing (one forced-choice forward pass in 8.3 s including warm-up; a batch of eight generations in 8.6 s) showed that the full design was feasible, so Family 3 and 4 were set at 40 items per cell and four paraphrases, as specified in the study plan. No pilot activation or behaviour value was used to choose any hypothesis, direction, measure or threshold.

## 9. Reference values fixed before the experiments

Derivation split: 11,126 texts; held-out split: 4,769 texts. Signature set sizes (aggregated over layers 14 to 22): anger 567, disgust 126, fear 592, happiness 677, sadness 320, surprise 1,334.

Reference overlap (Jaccard, derivation split), the predictor in H3.2 and H3.3:

| | anger | disgust | fear | happiness | sadness | surprise |
|---|---|---|---|---|---|---|
| anger | 1.000 | 0.068 | 0.003 | 0.000 | 0.020 | 0.011 |
| disgust | 0.068 | 1.000 | 0.011 | 0.000 | 0.028 | 0.000 |
| fear | 0.003 | 0.011 | 1.000 | 0.004 | 0.054 | 0.001 |
| happiness | 0.000 | 0.000 | 0.004 | 1.000 | 0.005 | 0.004 |
| sadness | 0.020 | 0.028 | 0.054 | 0.005 | 1.000 | 0.000 |
| surprise | 0.011 | 0.000 | 0.001 | 0.004 | 0.000 | 1.000 |

Pairs ranked by reference overlap (highest first): anger and disgust (0.068); fear and sadness (0.054); disgust and sadness (0.028); anger and sadness (0.020); disgust and fear (0.011); anger and surprise (0.011); happiness and sadness (0.005); happiness and surprise (0.004); fear and happiness (0.004); anger and fear (0.003); fear and surprise (0.001); sadness and surprise (0.000); disgust and surprise (0.000); disgust and happiness (0.000); anger and happiness (0.000).

Note recorded before the experiments: in the English derivation map the happiness and surprise overlap (0.004) is much weaker than in the Chinese full-data map, whereas anger and disgust (0.068) and fear and sadness (0.054) remain the two highest pairs. Family 1 and Family 2 still include the happiness and surprise pair, as preregistered in the design; its weak English overlap is a reason to expect smaller effects for that pair, and per-pair results are reported.

## 10. Code

`derive_map.py`, `run_experiments.py`, `analyse_experiments.py`, `stimuli/f2_disambiguation.py`, `stimuli/f3_f4_generation.py`, `stimuli/f1_items.csv`. The analysis code was checked on synthetic data with no effects (all 14 tests non-significant after Holm correction; `results/selftest_null_summary.txt`) and with an injected effect for every hypothesis (all 14 detected; `selftest_positive.py`, `results/selftest_positive_summary.txt`).
