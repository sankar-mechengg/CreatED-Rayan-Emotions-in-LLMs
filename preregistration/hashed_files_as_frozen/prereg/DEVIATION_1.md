# Deviation 1 from the preregistration: correction of the internal measure

Written after Families 1 to 3 had been run and analysed with the preregistered measure, and before any experimental result had been computed with the corrected measure defined here. The SHA-256 hash of this file and a UTC timestamp are recorded in `prereg/HASHES_DEVIATION_1.txt`.

## What was found

The preregistered internal measure scored a token by the mean of its signed, standardised activations over the signature neurons of each emotion. The signature neurons, however, are defined by how often a neuron falls in the top 5% of activations by absolute value. A neuron whose large-magnitude activations are negative is "active" in the signature sense when its signed value is low, so the signed mean can cancel or invert the signal. An exploratory validity check on the English held-out texts (baseline data only, outside the experiments; `diagnostics_measure.py`) confirmed this: the preregistered measure separated the six emotions at chance (six-way accuracy 0.165, majority 0.302; within-pair AUC 0.44 to 0.51), whereas two measures consistent with the signature definition separated them well above chance (six-way accuracy 0.425 and 0.429; within-pair AUC 0.68 to 0.84). The preregistered measure therefore does not measure what the hypotheses refer to.

Disclosure: before this check, the confirmatory analyses of Families 1, 2 and 3 had been run with the preregistered measure, and their results had been seen. Those results are reported in full as the preregistered results.

## Corrected measure (fixed here, before use)

* Primary corrected measure (mask fraction): at each preregistered position (stimulus span, final prompt token, generated span, planning state), the pooled activation of each of the nine layers is binarised with the same top-5% absolute-value rule used to define the signatures; the score of emotion e is the fraction of e's derivation-split signature neurons (aggregated over layers) that are active. The pair axis is the difference of two such scores. Pooling is the mean over the span, as in the baseline.
* Robustness measure (absolute z): the mean, over e's signature neurons, of the absolute pooled activation standardised by the derivation-split mean and standard deviation of the absolute pooled activation.

Everything else is unchanged: hypotheses, directions, positions, items, models, exclusions, the Holm family of 14 tests and the interpretation rule. The manuscript reports (a) the preregistered measure, (b) the corrected measure as the principal result, labelled as a deviation, and (c) the robustness measure.
