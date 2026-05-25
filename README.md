# VAPD: Vertical-Adapted Pruning and Distillation

> Official repository for the paper  
> **"From One to Many: Efficient Production and Deployment of Vertical LLM Families via Pruning and Distillation"**

VAPD is a **One-to-Many** framework that derives lightweight, hardware-optimized LLM variants from a single vertical foundation model via structured pruning and knowledge distillation. It eliminates the prohibitive cost of per-scale Continued Pre-training (CPT), enabling industrial-scale production of vertical-domain LLM families.

Since July 2025, VAPD has powered Sangfor Technologies' production XDR through five variants (0.8B–7B), serving millions of daily requests across cloud, on-premise, and edge environments.

---

## 🔑 Key Results

- **97% reduction** in marginal derivation cost per variant (2,747 → 81 GPU-hours)
- **61.4% reduction** in total model family cost (14,412 → 5,567 GPU-hours)
- **5 production variants** deployed in Sangfor Athena XDR (0.8B / 1.5B / 3B / 4.5B / 7B)
- VAPD-derived 3B model recovers **92% of teacher performance** with only **41% of the parameters**

---

## 📚 Repository Contents

### Documentation
- [`docs/domain_background.md`](docs/domain_background.md) — NTD domain background, attack taxonomy, and task formulations
- [`docs/data_details.md`](docs/data_details.md) — Full dataset composition (CPT / SFT / KD / calibration)
- [`docs/evaluation.md`](docs/evaluation.md) — Evaluation protocol and metrics
- [`docs/additional_results.md`](docs/additional_results.md) — Supplementary experimental results (Mistral-4.5B variant)
- [`docs/medical_evaluation.md`](docs/medical_evaluation.md) — Cross-domain evaluation on MLEC-QA
- [`docs/deployment.md`](docs/deployment.md) *(coming soon)* — Production deployment details

### Case Studies
- [`case_studies/vapd_vs_baseline.md`](case_studies/vapd_vs_baseline.md) — Qualitative comparison: VAPD vs. CPT+SFT vs. SFT on a Command Injection instance
- [`case_studies/encrypted_webshell.md`](case_studies/encrypted_webshell.md) — Encrypted Webshell detection in Sangfor Athena XDR

### Code *(coming soon)*
- `vapd/pruning/` — Taylor importance estimation, dependency-aware C→Dh pruning, mixed-domain calibration
- `vapd/distillation/` — α-mixed KL + CE distillation loss
- `vapd/evaluation/` — ATD and ARPD evaluation metrics
- `scripts/` — End-to-end training and evaluation entry points
- `configs/` — Per-variant pruning and training configurations

### Data Templates
- `data/templates/` *(coming soon)* — ATD / ARPD prompt schemas
- `data/samples/` *(coming soon)* — De-identified sample inputs and outputs

---

## 🚀 Quick Start

> 🚧 Code and detailed instructions are being released progressively. Check back for updates.

### Installation
```bash
# coming soon
```

### Pruning a 7B model to 3B
```bash
# coming soon
```

### Evaluation on NTD
```bash
# coming soon
```

---

## 🏗️ Method Overview

VAPD compresses a domain-adapted teacher model through three coordinated stages:

1. **Domain-adaptive importance estimation** — Gradient-based Taylor scores computed on a **3:1 mixture of domain-specific and general-domain calibration data**, preserving sparse vertical knowledge that magnitude-based metrics overlook.

2. **Dependency-aware intra-layer pruning** — A `C → D_h` pruning sequence that first reduces the LayerNorm feature dimension `C` (the global feature backbone), then the MLP intermediate dimension `D_h` within the stabilized manifold.

3. **Adaptive knowledge distillation** — A hybrid KL + CE objective with domain-tunable weight `α`, enabling the student to inherit the teacher's logit distribution while remaining anchored to ground truth.

See Section 3 of the paper for the full methodology.

---

## ⭐ Support

If you find VAPD useful, please consider giving the repo a star — it helps others discover the work.

---

## 📝 License

This project is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgments

This work was completed during Cong Ming's internship at Sangfor Technologies. We thank the Sangfor Athena XDR engineering team for production deployment support.