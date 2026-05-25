# Medical Domain Evaluation

This document details the cross-domain evaluation that demonstrates VAPD's generalization beyond cybersecurity. It expands on Section 4.2.2 of the paper.

## 1. Motivation

A natural question for any vertical-domain compression framework is whether its benefits are specific to the calibration domain. To test cross-domain generality, we apply the **identical VAPD pipeline**—pruning under domain-mixed calibration, the C→D_h ordering, and adaptive distillation—to a second vertical: **medical question answering**.

The only modification across domains is the distillation hyperparameter `α`, which is re-tuned per domain (`α = 1.0` for NTD, `α = 0.75` for medical; see Section 4.3.2 of the paper).

## 2. Benchmark: MLEC-QA

We use **MLEC-QA**, curated from the **National Medical Licensing Examination in China (NMLEC)**. The benchmark spans five medical subdisciplines:

| Subset | Full Name |
|--------|-----------|
| Clinic | Clinical Medicine |
| CWM | Traditional Chinese Medicine Combined with Western Medicine |
| PublicHealth | Public Health |
| Stomatology | Stomatology (Dentistry) |
| TCM | Traditional Chinese Medicine |

## 3. Evaluation Setup

To form our evaluation benchmark, we **randomly sample 10% of each subset's test set**, yielding 1,362 questions in total. All evaluations are performed under a **zero-shot** methodology.

| Dataset | Test Set Original Size | Sample Size |
|---------|------------------------|-------------|
| MLEC-QA Clinic | 3,362 | 336 |
| MLEC-QA CWM | 2,674 | 268 |
| MLEC-QA PublicHealth | 1,853 | 185 |
| MLEC-QA Stomatology | 2,644 | 264 |
| MLEC-QA TCM | 3,086 | 309 |
| **Total** | **13,619** | **1,362** |

## 4. Medical Teacher Model

Following the **DISC protocol** (Bao et al., 2023), we train a medical teacher via CPT + SFT on its public dataset, then apply VAPD to derive 3B and 1.5B variants.

## 5. Results

Quantitative results are reported in **Table 4 of the paper**. Briefly, VAPD-compressed Mistral variants retain solid performance across all five subsets:

| Model | Avg. Across 5 Subsets |
|-------|----------------------|
| Mistral-7B (Teacher) | 63.60% |
| Mistral-3B (VAPD) | 60.37% |
| Mistral-1.5B (VAPD) | 56.76% |

## 6. Takeaway

The medical evaluation confirms that VAPD is **not overfit to cybersecurity**. The same pipeline transfers cleanly to a domain with markedly different characteristics:

- NTD favors **structured, rule-like reasoning** over short HTTP payloads → `α = 1.0` (pure KL inheritance)
- Medical QA involves **factual recall** over longer free-text questions → `α = 0.75` (KL + CE hybrid)

This dual evidence—different domain, different optimal hyperparameter, same pipeline—supports VAPD's positioning as a general vertical-domain compression framework rather than a cybersecurity-specific technique.

---

## Related Documents

- 🌐 [`domain_background.md`](domain_background.md) — NTD domain context
- 📊 [`data_details.md`](data_details.md) — Cybersecurity dataset details
- 🧬 [`additional_results.md`](additional_results.md) — Mistral 4.5B variant results