# Case Study: VAPD vs. Baseline Output Comparison

This case study illustrates the qualitative advantages of VAPD over conventional adaptation pipelines (CPT+SFT and SFT) on the ARPD task. It expands on the results in Section 4.2 of the paper with a representative real-world example.

## Test Instance

We present a representative example from the ARPD task. The input is a complete HTTP transaction log involving a **Command Injection** attack, with the ground-truth attack result labeled `SUCCESS`.

<p align="center">
  <img src="../docs/figures/arpd_example.png" width="700">
  <br>
  <em>Figure 1: The ARPD test instance used in this case study—a Command Injection attack with <code>SUCCESS</code> ground truth. The malicious payload <code>system('whoami')</code> is injected into the <code>id</code> parameter; the response body contains <code>www-data</code>, the typical web-server user identity returned by the <code>whoami</code> command on Linux, evidencing successful command execution. (Sensitive information has been removed.)</em>
</p>

## Output Comparison

We compare the outputs from three models at the same parameter scale (3B):
- **VAPD (Ours)** — pruned from a 7B vertical teacher and recovered via knowledge distillation
- **CPT+SFT** — continued pre-training followed by supervised fine-tuning at the 3B scale
- **SFT** — supervised fine-tuning on the 3B base model without CPT

<p align="center">
  <img src="figures/output_comparison.png" width="750">
  <br>
  <em>Figure 2: Side-by-side output comparison of VAPD, CPT+SFT, and SFT on the same Command Injection instance. Annotations use green ✓ to mark correct content (label, payload identification) and red ✗ to mark errors (wrong label, wrong attack type, severe hallucination of a non-existent response header).</em>
</p>

## Analysis

### ✅ VAPD (Ours): Correct and Faithful

The VAPD-compressed model correctly identifies the attack type as **Command Injection** and accurately predicts the result as **SUCCESS**. Critically, its explanation is **grounded in the actual HTTP response**—it references the presence of `'www-data'` in the response body as evidence of successful command execution. The reasoning chain is coherent, factually consistent, and directly traceable to observable evidence in the log.

### ⚠️ CPT+SFT: Correct Label, Hallucinated Explanation

The CPT+SFT model also predicts `SUCCESS` and correctly identifies the payload `system('whoami')`—but its explanation **mixes correct observations with fabricated evidence**:

1. **Fabricated evidence:** It falsely claims the existence of a response header `'X-Command-Status: Executed'`, which **does not appear anywhere in the original log**.
2. **Overreach in inference:** It asserts "full command execution" based solely on the `whoami` output, an unwarranted generalization from a single benign command to arbitrary code execution.

> 🚨 **Operational risk:** In real-world security operations, such hallucinations constitute **critical failures**. They risk misleading analysts during incident triage and may trigger erroneous automated containment responses (e.g., quarantining an asset based on fabricated evidence). A correct label with a wrong explanation is in some respects worse than an incorrect label, because it erodes trust in subsequent model outputs.

### ❌ SFT: Misclassification and Wrong Outcome

The SFT model fails on **both** dimensions evaluated:
- Misclassifies the attack type as **XSS** (it is Command Injection)
- Predicts the result as **FAILURE** (it is SUCCESS)
- Misinterprets the `www-data` response as "no visible script execution," failing to recognize it as evidence of `whoami` having executed server-side

This reflects poor semantic understanding of HTTP payload structure and weak generalization under limited supervised data—precisely the gap that motivates domain-adaptive pre-training in the conventional pipeline, and that VAPD bridges through structural knowledge inheritance from the teacher.

## Takeaways

This case study highlights three properties of VAPD that aggregate metrics in the paper cannot fully convey:

1. **Reasoning fidelity is preserved at compressed scale.** VAPD's 3B variant produces explanations as faithful as the 7B teacher, while same-scale CPT+SFT models hallucinate.
2. **Structural inheritance protects against ungrounded inference.** By inheriting the teacher's parameter structure rather than learning the task from scratch, the student retains the teacher's grounded reasoning behavior.
3. **Aggregate accuracy can hide qualitative failures.** CPT+SFT and VAPD may both produce the correct top-level label, but their downstream operational value differs dramatically.

For full quantitative comparisons across all metrics and scales, see Tables 2 and 3 of the paper.