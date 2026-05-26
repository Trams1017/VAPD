# Copyright 2026 VAPD authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
VAPD: Vertical-Adapted Pruning for Industrial LLM Families.

This module implements the structured pruning stage of VAPD
(Section 3.2 of the paper). The pruner operates on a domain-adapted
foundation model and derives lightweight, hardware-optimized variants
by selectively removing parameters along the LayerNorm channel
dimension C and the MLP intermediate dimension D_h, using gradient-
based importance estimates computed on a domain-aware calibration
distribution.

Three design choices distinguish VAPD's pruning from generic
structured-pruning baselines:

  (1) Importance is estimated via first-order Taylor expansion
      (Eq. 3), which decouples sensitivity from weight magnitude
      and allows low-magnitude but domain-critical neurons to be
      preserved (Section 3.2.2).

  (2) Calibration data is constructed as a 3:1 mixture of domain
      and general instances, which shifts the gradient distribution
      so that domain-sensitive parameters receive amplified scores
      relative to broadly distributed linguistic parameters
      (Section 3.2.2, Table 9).

  (3) The order of pruning across structural dimensions follows
      a dependency-aware sequence C → head_dim → D_h, which first
      stabilizes the global feature manifold before optimizing local
      transformation capacity within the reduced subspace
      (Section 3.2.3, Table 6).
"""

import gc
import argparse
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import DataCollatorForSeq2Seq
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm, Qwen2RotaryEmbedding,
)
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner import torch_pruning as tp

from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_dataset
from llamafactory.extras.constants import IGNORE_INDEX


# Target architecture for derived variants. The default below derives
# a 3B student from a Qwen-2.5-7B teacher, matching the official
# Qwen-2.5-3B geometry for direct comparison against the conventional
# CPT+SFT pipeline (paper Table 2). Override for hardware-adaptive
# custom sizes such as 4.5B (paper Section 5.2).
TARGET_ARCHITECTURE = dict(
    embed=2240,       # hidden_size  — LayerNorm channel dim C
    mlp=11032,        # intermediate_size — MLP local capacity D_h
    qhead=28,
    kvhead=4,
    head_dim=128,
    n_layers=28,
)

# Recommended pruning sequence (Section 3.2.3). The first dimension
# defines the global feature manifold and is pruned first; subsequent
# dimensions optimize local capacity within the stabilized subspace.
VAPD_ORDER = ("embed", "head_dim", "mlp")

# Custom group-pruning rules for the RMSNorm variants used by the
# Qwen-2.5 and Mistral families.
RMSNORM_PRUNERS = {
    Qwen2RMSNorm: llama_pruner.hf_rmsnorm_pruner,
    MistralRMSNorm: llama_pruner.hf_rmsnorm_pruner,
}


class VAPDPruner:
    """
    Vertical-Adapted Pruner.

    Encapsulates the three design choices outlined in this module's
    header docstring into a single pipeline that takes a domain-adapted
    teacher and returns a structurally pruned student model.
    """

    def __init__(self, teacher_model, calibration_batch,
                 target_architecture=TARGET_ARCHITECTURE,
                 importance_metric="taylor",
                 pruning_order=VAPD_ORDER,
                 trace_input_shape=(4, 20)):
        self.teacher = teacher_model
        self.calibration_batch = calibration_batch
        self.target = target_architecture
        self.metric = importance_metric
        self.order = pruning_order
        self.trace_input_shape = trace_input_shape

        # Read the teacher's current structural dimensions; these
        # serve as the source for each per-dimension pruning ratio.
        cfg = teacher_model.config
        self.source = dict(
            embed=cfg.hidden_size,
            mlp=cfg.intermediate_size,
            qhead=cfg.num_attention_heads,
            kvhead=cfg.num_key_value_heads,
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            n_layers=cfg.num_hidden_layers,
        )

    # ------------------------------------------------------------------
    # Importance estimation
    # ------------------------------------------------------------------

    def _estimate_first_order_sensitivity(self, model):
        """
        Compute per-parameter Taylor sensitivity scores on the
        calibration distribution (Eq. 3).

        The masked next-token cross-entropy is differentiated through
        the model in a single backward pass; the resulting `.grad`
        tensors, multiplied with the parameter values themselves,
        give the per-parameter importance scores. Group aggregation
        (Eq. 5) is delegated to the structural pruner via the
        `TaylorImportance` operator.

        Because the calibration batch is drawn from the 3:1 mixed
        distribution P_mix = ¾·P_vert + ¼·P_gen, gradients of
        parameters sensitive to the vertical distribution are
        amplified relative to those of broadly distributed linguistic
        parameters — even at low numerical magnitude. This is the
        mechanism by which sparse, domain-critical neurons are
        shielded from removal (Section 3.2.2).
        """
        model.zero_grad()
        batch = {k: v.to(model.device) for k, v in self.calibration_batch.items()}

        outputs = model(**batch)
        shift_logits = outputs["logits"][..., :-1, :]
        shift_labels = batch["labels"][..., 1:]
        loss_mask = (shift_labels != IGNORE_INDEX).flatten()

        flat_logits = shift_logits.contiguous().view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.contiguous().view(-1)

        per_token_ce = torch.nn.functional.cross_entropy(
            flat_logits, flat_labels, reduction="none"
        )
        loss = (per_token_ce * loss_mask).sum() / loss_mask.sum()
        loss.backward()

    def _build_importance_operator(self):
        """
        Construct the group-level importance operator that the
        structural pruner uses to rank coupled parameter groups.

        Taylor is the VAPD default; the alternatives are retained
        primarily to reproduce the importance-metric ablation
        reported in Table 8 of the paper.
        """
        if self.metric == "taylor":
            # Group reduction by summation, motivated by the linear
            # superposition of first-order Taylor effects (Eq. 5).
            return llama_pruner.TaylorImportance(
                group_reduction="sum", taylor="param_first"
            )
        if self.metric == "l1":
            return llama_pruner.MagnitudeImportance(p=1)
        if self.metric == "l2":
            return llama_pruner.MagnitudeImportance(p=2)
        if self.metric == "random":
            return tp.importance.RandomImportance()
        raise ValueError(f"unknown importance metric: {self.metric}")

    # ------------------------------------------------------------------
    # Dimension-wise structured pruning
    # ------------------------------------------------------------------

    def _prune_dimension(self, model, pruner, dim_name):
        """
        Prune a single structural dimension to its target size.

        For Taylor-based pruning, gradients are recomputed before each
        dimension's step so that importance scores reflect the
        partially-pruned model state — this is what distinguishes
        VAPD's ordered pruning from one-time pruning, where gradients
        are computed only once at the start (Table 6, row 3).
        """
        if self.source[dim_name] == self.target[dim_name]:
            return

        if self.metric == "taylor":
            self._estimate_first_order_sensitivity(model)

        if dim_name == "embed":
            # The LayerNorm channel C is the global feature backbone:
            # pruning it forces all adjacent projection matrices to
            # realign to a compact, informative subspace.
            roots = [model.model.norm]
        elif dim_name == "head_dim":
            # Pruning head_dim through k_proj / v_proj reduces the
            # per-head representation while preserving the head count.
            roots = (
                [layer.self_attn.k_proj for layer in model.model.layers] +
                [layer.self_attn.v_proj for layer in model.model.layers]
            )
        elif dim_name == "mlp":
            # Pruning D_h reduces local transformation capacity within
            # each layer. When applied after C, this step optimizes
            # local capacity against an already-stabilized manifold.
            roots = [layer.mlp.gate_proj for layer in model.model.layers]
        else:
            raise ValueError(f"unknown structural dimension: {dim_name}")

        ratio = 1.0 - self.target[dim_name] / self.source[dim_name]
        pruner.current_step = 0
        pruner.per_step_ch_sparsity = [0, ratio]
        pruner.root_instances = roots
        pruner.step()

    def _finalize_architecture(self, model):
        """
        Synchronize the model's config and runtime modules with the
        new structural dimensions, and truncate the layer stack if
        the target depth is shallower than the source.
        """
        if self.target["n_layers"] < self.source["n_layers"]:
            model.model.layers = nn.ModuleList(
                model.model.layers[: self.target["n_layers"]]
            )

        model.config.hidden_size = self.target["embed"]
        model.config.intermediate_size = self.target["mlp"]
        model.config.num_hidden_layers = self.target["n_layers"]
        model.config.head_dim = self.target["head_dim"]

        # Rotary positional embeddings are tied to head_dim and must
        # be rebuilt for the new shape.
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            attn.head_dim = self.target["head_dim"]
            attn.layer_idx = i
            attn.rotary_emb = Qwen2RotaryEmbedding(
                self.target["head_dim"],
                attn.max_position_embeddings,
                attn.rope_theta,
            ).to(model.device)

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self):
        """
        Execute the VAPD pruning pipeline end-to-end and return the
        pruned model. The teacher is left unmodified.
        """
        gc.collect()
        torch.cuda.empty_cache()

        model = deepcopy(self.teacher)
        for p in model.parameters():
            p.requires_grad_(True)

        trace_input = torch.ones(
            self.trace_input_shape, dtype=torch.long, device=model.device,
        )
        pruner = tp.pruner.MetaPruner(
            model, trace_input,
            importance=self._build_importance_operator(),
            customized_pruners=RMSNORM_PRUNERS,
            root_module_types=[], root_instances=[],
        )

        # Dependency-aware ordered pruning (Section 3.2.3).
        for dim_name in self.order:
            self._prune_dimension(model, pruner, dim_name)

        self._finalize_architecture(model)
        return model


# ----------------------------------------------------------------------
# Command-line entry
# ----------------------------------------------------------------------

def _load_teacher_and_calibration(model_path, dataset_name, batch_size,
                                  dataset_dir="data/calibration"):
    """
    Load the domain-adapted teacher model and assemble one batch from
    the calibration dataset. The dataset is expected to follow the
    3:1 domain-general mixture described in Section 3.2.2.
    """
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(dict(
        stage="sft",
        model_name_or_path=model_path,
        dataset=dataset_name,
        dataset_dir=dataset_dir,
        template="default",
        cutoff_len=8192,
        output_dir="dummy_dir",
    ))
    tok_module = load_tokenizer(model_args)
    tokenizer = tok_module["tokenizer"]
    trainset = get_dataset(
        model_args, data_args, training_args, "sft", **tok_module
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX
    )
    teacher = load_model(
        tokenizer, model_args, finetuning_args, is_trainable=False
    )
    calibration_batch = next(iter(DataLoader(
        trainset, batch_size, shuffle=False, collate_fn=collator,
    )))
    return teacher, tokenizer, calibration_batch


def main():
    parser = argparse.ArgumentParser(
        description="Derive a VAPD-pruned student from a vertical teacher."
    )
    parser.add_argument(
        "--model_name_or_path",
        default="<path to the domain-adapted teacher model>",
        help="Teacher checkpoint obtained from CPT + SFT on the target domain.",
    )
    parser.add_argument(
        "--dataset", default="<calibration dataset name>",
        help="Calibration dataset registered with LLaMA-Factory's data loader. "
             "Should follow the 3:1 domain-general mixture (Section 3.2.2).",
    )
    parser.add_argument(
        "--importance", default="taylor",
        choices=["taylor", "l1", "l2", "random"],
        help="Importance metric for ranking parameter groups (Table 8).",
    )
    parser.add_argument(
        "--order", default=",".join(VAPD_ORDER),
        help="Comma-separated dimension order for sequential pruning. "
             "VAPD default: embed,head_dim,mlp (Table 6).",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=20)
    args = parser.parse_args()

    teacher, tokenizer, calibration_batch = _load_teacher_and_calibration(
        args.model_name_or_path, args.dataset, args.batch_size,
    )

    pruner = VAPDPruner(
        teacher_model=teacher,
        calibration_batch=calibration_batch,
        target_architecture=TARGET_ARCHITECTURE,
        importance_metric=args.importance,
        pruning_order=tuple(s.strip() for s in args.order.split(",")),
        trace_input_shape=(args.batch_size, args.seq_length),
    )
    student = pruner.run()

    save_dir = (
        f"{args.model_name_or_path}-vapd"
        f"-c{TARGET_ARCHITECTURE['embed']}"
        f"-h{TARGET_ARCHITECTURE['mlp']}"
        f"-n{TARGET_ARCHITECTURE['n_layers']}"
        f"-{args.importance}"
    )
    student.save_pretrained(save_directory=save_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(save_dir)
    print(f"VAPD-derived student saved to: {save_dir}")
    print(f"Pruning order applied: {' → '.join(pruner.order)}")


if __name__ == "__main__":
    main()