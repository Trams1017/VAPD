# Copyright 2026 VAPD authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
VAPD: Domain-Adaptive Knowledge Distillation for Pruned Vertical LLMs.

This module implements the capability-recovery stage of VAPD
(Section 3.3 of the paper). After structural pruning has reduced the
model's parameter count, knowledge distillation is used to restore
task performance under the new architectural budget. The teacher is
the pre-pruning domain-adapted foundation model; the student is the
VAPD-pruned variant produced by `prune.py`.

The distillation objective combines two complementary forces:

  (1) A forward KL term aligns the student's output distribution with
      the teacher's full predictive distribution, transferring the
      structured logit patterns acquired through domain pre-training
      (Eq. 4).

  (2) A standard cross-entropy term anchors the student to ground-
      truth supervision, mitigating the risk that any uncertainty or
      hallucination in the teacher's outputs propagates undiluted
      into the smaller student.

These are blended via a single domain-tunable hyperparameter α in
the composite objective L_total = α·L_KL + (1−α)·L_CE (Eq. 5). The
optimal α is task-dependent: α=1.00 for structured-reasoning tasks
such as network traffic detection, where full inheritance of the
teacher's logit distribution is desirable; α=0.75 for factual-recall
domains such as medical QA, where ground-truth anchoring guards
against teacher hallucination (Section 4.3.2, Table 7).
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer

from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import get_train_args


# Default α for the composite distillation objective. α=1.00 reproduces
# the NTD configuration reported in the paper (Section 4.3.2); for
# factual-recall domains we recommend α=0.75 as a starting point.
ALPHA_DEFAULT = 1.0


class VAPDDistillationTrainer(Seq2SeqTrainer):
    """
    Trainer that recovers the capability of a VAPD-pruned student
    by distilling from its pre-pruning teacher under the composite
    objective of Eq. (5).

    The teacher is held frozen and evaluated in lock-step with each
    student forward pass. The composite loss is computed in-place
    inside `compute_loss`, replacing the default cross-entropy loss
    of the parent Seq2SeqTrainer.
    """

    def __init__(self, *args, teacher_model=None, alpha=ALPHA_DEFAULT, **kwargs):
        super().__init__(*args, **kwargs)
        if teacher_model is None:
            raise ValueError(
                "VAPDDistillationTrainer requires a teacher_model. "
                "Pass the pre-pruning domain-adapted foundation model."
            )

        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the composite distillation loss of Eq. (5).

        The student is run with gradient tracking enabled; the teacher
        is run under torch.no_grad() to avoid memory and compute
        overhead for parameters that will never be updated. Both
        models share the same input batch, ensuring that the KL term
        operates on aligned predictive distributions.
        """
        # Student forward pass — provides both the student logits
        # for the KL term and the cross-entropy loss against the
        # ground-truth labels for the CE term.
        student_outputs = model(**inputs)
        ce_loss = student_outputs.loss
        student_logits = student_outputs.logits

        # Teacher forward pass in lock-step. We detach the teacher's
        # graph entirely; the teacher exists only as a fixed target
        # distribution during student training.
        with torch.no_grad():
            teacher_logits = self.teacher(**inputs).logits

        # Forward KL divergence D_KL(p_t || p_s) — Eq. (4).
        #
        # We use log-softmax on the student side because PyTorch's
        # F.kl_div expects log-probabilities for its input argument.
        # The `batchmean` reduction normalizes by batch size, which
        # places L_KL on the same magnitude scale as L_CE so that α
        # remains an interpretable mixing weight rather than a
        # rescaling factor.
        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )

        # Composite objective — Eq. (5).
        loss = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss

        return (loss, student_outputs) if return_outputs else loss


def _load_student_and_teacher(student_path, teacher_path, dataset, output_dir,
                              learning_rate, num_epochs, batch_size, grad_accum):
    """
    Load the VAPD-pruned student and the pre-pruning teacher. The
    student is loaded in trainable mode; the teacher will be frozen
    by `VAPDDistillationTrainer.__init__`.
    """
    student_args, _, training_args, student_ft_args, _ = get_train_args(dict(
        stage="sft",
        model_name_or_path=student_path,
        dataset=dataset,
        template="default",
        cutoff_len=8192,
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
    ))
    tok_module = load_tokenizer(student_args)
    tokenizer = tok_module["tokenizer"]
    student = load_model(
        tokenizer, student_args, student_ft_args, is_trainable=True
    )

    teacher_args, _, _, teacher_ft_args, _ = get_train_args(dict(
        stage="sft",
        model_name_or_path=teacher_path,
        dataset=dataset,
        template="default",
        cutoff_len=8192,
        output_dir="dummy",
    ))
    teacher = load_model(
        tokenizer, teacher_args, teacher_ft_args, is_trainable=False
    )
    return student, teacher, tokenizer, training_args


def main():
    parser = argparse.ArgumentParser(
        description="Recover a VAPD-pruned student via domain-adaptive distillation."
    )
    parser.add_argument(
        "--student", default="<path to the VAPD-pruned student>",
        help="Output of `prune.py`. Loaded in trainable mode.",
    )
    parser.add_argument(
        "--teacher", default="<path to the domain-adapted teacher>",
        help="The pre-pruning foundation model used as the distillation target.",
    )
    parser.add_argument(
        "--dataset", default="<distillation dataset name>",
        help="Same domain dataset used for the teacher's SFT stage.",
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA_DEFAULT,
        help="Distillation mixing weight in Eq. (5). "
             "NTD: 1.00; medical: 0.75. See paper Section 4.3.2.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--output_dir", default="./vapd_kd_output")
    args = parser.parse_args()

    student, teacher, tokenizer, training_args = _load_student_and_teacher(
        student_path=args.student,
        teacher_path=args.teacher,
        dataset=args.dataset,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )

    # The trainer below replaces the standard SFT trainer in
    # LLaMA-Factory's `run_sft` flow. Dataset construction and the
    # data collator follow the standard SFT pipeline and are omitted
    # here; the VAPD-specific behavior is fully contained in
    # `VAPDDistillationTrainer.compute_loss`.
    trainer = VAPDDistillationTrainer(
        model=student,
        teacher_model=teacher,
        alpha=args.alpha,
        args=training_args,
        tokenizer=tokenizer,
    )
    print(f"Configured VAPDDistillationTrainer with α = {args.alpha}")
    print("Plug into the SFT data pipeline and call `trainer.train()`.")


if __name__ == "__main__":
    main()