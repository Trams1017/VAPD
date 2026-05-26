# VAPD Reference Code

This directory contains the reference implementation of VAPD's two
core stages — structured pruning and knowledge distillation. Each
stage is a single self-contained file:

| File | What it implements | Maps to |
|------|--------------------|---------|
| [`prune.py`](prune.py) | Taylor importance + dimension-ordered structured pruning (C → head_dim → D_h) | Paper Section 3.2 |
| [`distillation_trainer.py`](distillation_trainer.py) | Hybrid α · L_KL + (1 − α) · L_CE trainer | Paper Section 3.3, Eq. (5) |

## Scope and Intent

These files are provided to make the methods **directly verifiable
from source**. They are not packaged as a runnable end-to-end
pipeline, for two reasons:

1. The **domain-adapted teacher model** and the **3:1 mixed
   calibration dataset** used in our experiments cannot be
   redistributed due to customer-data and operational security
   constraints (see `docs/data_details.md`).
2. The surrounding infrastructure — data loading, dataset
   registration, distributed training launch — is the standard
   [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) SFT
   pipeline, which is already open-source. Reproducing it here
   would add bulk without adding clarity.

In both files, the lines that constitute VAPD's actual contribution
are flagged with `# === VAPD CORE: ... ===` comment blocks, with
cross-references to the corresponding paper equations.

## Reproducing the Ablations

The CLI flags in `prune.py` expose the two ablation axes reported in
the paper:

```bash
# VAPD default
python prune.py --importance taylor --order embed,head_dim,mlp

# D_h → head_dim → C ablation
python prune.py --importance taylor --order mlp,head_dim,embed

# Importance ablation
python prune.py --importance random
python prune.py --importance l1
python prune.py --importance l2
```

The α ablation for distillation is exposed via the
`--alpha` flag in `distillation_trainer.py`.

## Dependencies

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — training framework
- `transformers==4.45.0` for Qwen-2.5; `transformers==4.43.1` for Mistral
- `torch >= 2.1`
