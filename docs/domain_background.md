# Network Traffic Detection (NTD): Domain Background

This document provides the domain context for the NTD task used to evaluate VAPD in our CIKM 2026 paper. It expands on Section 4.1.1 of the paper with details on the cybersecurity setting, attack taxonomy, task formulations, and production deployment context.

## 1. Why LLMs for Network Traffic Detection

Network Traffic Detection (NTD) constitutes a fundamental layer of enterprise cybersecurity, aimed at identifying malicious activities within network flows—such as SQL injection, cross-site scripting (XSS), and command injection—before they compromise infrastructure.

Traditional Intrusion Detection Systems (IDS) primarily rely on **signature-based matching** (e.g., regular expressions, Snort rules, YARA patterns). While computationally efficient, these approaches face fundamental limitations against modern adversarial tactics:

- **Traffic encryption** (TLS 1.3, encrypted SNI) obscures payload contents from rule engines that depend on plaintext inspection.
- **Payload obfuscation** (URL encoding, Base64 nesting, polymorphic shells) trivially bypasses static patterns.
- **Zero-day attacks** lack prior signatures, leading to high false-negative rates on novel threats.
- **Rule maintenance** becomes economically infeasible as attack variants grow combinatorially.

Large Language Models (LLMs) offer a paradigm shift from **pattern matching to semantic analysis**. By treating HTTP requests and responses as linguistic sequences, LLMs capture deep contextual dependencies and identify anomalous patterns that deviate from benign protocol semantics.

### A Concrete Example

Consider a Command Injection attack where the malicious payload is obfuscated to evade signature detection. A signature-based IDS may miss the request itself, but the **server response** reveals the attack succeeded: it returns the string `'www-data'` (the typical web-server user identity returned by the `whoami` command on Linux). An LLM that has learned the *semantics* of HTTP transactions can:

1. Recognize `'www-data'` in the response body as evidence of command execution as the web-server user,
2. Cross-reference the request payload with the response content to infer attack outcome,
3. Produce a grounded explanation traceable to observable evidence in the log.

This semantic, evidence-grounded reasoning is precisely what static rules cannot achieve. A worked example is provided in [`../case_studies/vapd_vs_baseline.md`](../case_studies/vapd_vs_baseline.md), where VAPD-derived models perform this reasoning faithfully while same-scale baselines fabricate evidence (hallucinate response headers that don't exist) despite reaching the same top-level label.

At Sangfor Technologies, integrating LLMs into NTD aims to construct an intelligent defense layer capable of detecting stealthy, evolving threats with high precision—overcoming the structural limitations of rule-based systems.

## 2. Why a Model Family Rather Than a Single Model

Production NTD must operate across radically different deployment environments:

| Tier | Environment | Typical Hardware | Latency Budget | VAPD Variant |
|------|-------------|------------------|----------------|--------------|
| Cloud SaaS | Centralized inspection backbone | NVIDIA A100 / A800 | Higher tolerance | 7B, 4.5B |
| On-premise | Customer data center appliance | RTX 4090 / 4080 | Moderate | 3B, 1.5B |
| Edge | Branch office / IoT gateway | RTX 3090 / 3080 | Strict | 0.8B |

A single model size cannot serve all three tiers. Open-source LLM families typically release at fixed scales (e.g., Qwen-2.5 at 1.5B / 3B / 7B), which leaves gaps for **hardware-specific budgets**. For example, a customer appliance with a specific GPU memory budget may be optimally served by a 4.5B variant—a size unavailable in any open-source release.

This motivates the **One-to-Many** production paradigm of VAPD: derive a family of hardware-optimized variants from a single domain-adapted teacher, rather than performing prohibitively expensive per-scale Continued Pre-training. Empirically, the 4.5B variant we derive from a 7B Mistral teacher reaches **94.36% Avg.**—within 0.15 points of the teacher's 94.50%—while delivering **2.125× higher single-node throughput**. See [`additional_results.md`](additional_results.md) for the full Mistral family scaling analysis.

## 3. Attack Taxonomy: 13 Categories

Our Attack Type Detection (ATD) task classifies HTTP requests into **13 categories** curated from production traffic at Sangfor Technologies. The taxonomy reflects Sangfor's internal classification system, refined through years of production telemetry.

| # | Attack Type | Training Samples | Test Samples |
|---|-------------|------------------|--------------|
| 1 | Java Injection | 14,414 | 150 |
| 2 | Java Deserialization | 12,021 | 125 |
| 3 | PHP Injection | 13,836 | 144 |
| 4 | SQL Injection | 11,252 | 117 |
| 5 | Webshell Upload | 10,771 | 112 |
| 6 | Webshell Encrypted | 9,617 | 100 |
| 7 | XSS Injection | 13,071 | 135 |
| 8 | Shiro Deserialization | 8,751 | 91 |
| 9 | Information Disclosure | 10,579 | 110 |
| 10 | Command Injection | 11,540 | 120 |
| 11 | Weak Password | 9,617 | 100 |
| 12 | Unauthorized Access | 9,617 | 100 |
| 13 | Directory Traversal | 14,415 | 150 |
| | **Total** | **149,501** | **1,554** |

The dataset spans **149,501 training samples and 1,554 test samples** across the 13 categories. The distribution is moderately imbalanced—reflecting the natural prevalence of attack types in production traffic—rather than artificially balanced. Frequencies range from ~5.9% (Shiro Deserialization) to ~9.6% (Directory Traversal, Java Injection), with no class dominating.

### Taxonomy Design Notes

A few aspects of this taxonomy reflect deliberate production choices rather than off-the-shelf OWASP categories:

- **Language-specific injection classes** (Java Injection, PHP Injection) are separated because their exploitation patterns, payload structures, and downstream remediation differ substantially in enterprise environments.
- **Deserialization vulnerabilities** are split into a general Java Deserialization class and a Shiro-specific class, reflecting Shiro's prevalence and distinct attack signatures in Chinese enterprise stacks.
- **Webshell** is decomposed into Upload (the act of placing the shell) and Encrypted (the use of encryption to evade detection on subsequent shell traffic)—two operationally distinct events that warrant separate handling in the SOC pipeline.
- **Weak Password** and **Unauthorized Access** are first-class categories because they account for a disproportionate share of real incidents despite being less technically sophisticated than injection attacks.

The Command Injection category illustrated in [`../case_studies/vapd_vs_baseline.md`](../case_studies/vapd_vs_baseline.md) is one representative class. For prompt templates, the broader SFT/KD dataset structure, and the CPT corpus composition, see [`data_details.md`](data_details.md).

## 4. Task Formulations

NTD at Sangfor is decomposed into two complementary tasks:

### 4.1 Attack Type Detection (ATD)

**Goal:** Classify the attack vector present in an HTTP request.

**Input:** An HTTP request (method, URL, headers, body), together with a system prompt `p`.
**Output:** A single attack-type label `u` drawn from the 13-category taxonomy.

Formally, given input `(p, r)` and label sequence `u`:

```
L_ATD(θ) = -E[(p,r),u]~D Σ log P_θ(u_t | u_<t, (p, r))
```

(Eq. 6 in the paper.)

### 4.2 Attack Result and Payload Detection (ARPD)

**Goal:** Beyond classification, judge whether the attack succeeded and extract the malicious payload(s).

**Input:** The HTTP request `r`, the server response `s`, and the previously detected attack label `ψ`.
**Output:** A structured sequence `v` containing:
- Attack outcome: `Success` / `Failure` / `Unknown`
- Extracted malicious payload(s) recovered from the request/response

Token-level weighting `w` emphasizes structured outcome tokens (Eq. 7):

```
L_ARPD(θ) = -E[(p,r,s,ψ),v]~D Σ w · log P_θ(v_t | v_<t, (p, r, s, ψ))
```

### 4.3 Why the Two-Stage Design

We split detection from outcome analysis because:

1. **Modularity.** ATD can run alone as a lightweight pre-filter; ARPD escalates only on detected attacks, reducing production load by orders of magnitude.
2. **Capability separation.** ATD is a classification problem solvable with the request alone; ARPD requires joint reasoning over request *and* response and is a generation task. Separating them allows targeted optimization and clean failure-mode analysis.
3. **Operational alignment.** Security analysts typically first ask "what attack is this?" and only then "did it succeed, and what was exfiltrated?" The two-stage pipeline mirrors this workflow.

## 5. Evaluation

Detailed evaluation protocol, test set composition, metrics, and example inputs are documented in [`evaluation.md`](evaluation.md).

Briefly: we evaluate on 1,554-sample test sets for both ATD (weighted precision/recall over 13 classes) and ARPD (three-class accuracy + payload recovery rate). All decoding is deterministic and greedy (temperature = 0) to eliminate evaluation randomness.

For an explanation of why aggregate metrics alone are insufficient for vertical-domain deployment—and how qualitative failures like hallucinated evidence can hide behind correct top-level labels—see [`evaluation.md §4`](evaluation.md#4-what-aggregate-metrics-cannot-capture) and the worked example in [`../case_studies/vapd_vs_baseline.md`](../case_studies/vapd_vs_baseline.md).

## 6. Production Context

NTD models derived via VAPD are integrated into **Sangfor Athena XDR** (Extended Detection and Response), serving millions of requests daily across cloud, on-premise, and edge deployments. Production performance metrics for each variant (detection rate, false positive rate, training cost) are reported in Table 10 of the paper.

---

## Related Documents

- 📊 [`data_details.md`](data_details.md) — Full dataset composition (CPT / SFT / KD / calibration)
- 🧪 [`evaluation.md`](evaluation.md) — Evaluation protocol and metrics
- 🧬 [`additional_results.md`](additional_results.md) — Supplementary results including the Mistral-4.5B variant
- 🏥 [`medical_evaluation.md`](medical_evaluation.md) — Cross-domain evaluation on MLEC-QA
- 🔍 [`../case_studies/vapd_vs_baseline.md`](../case_studies/vapd_vs_baseline.md) — VAPD vs. CPT+SFT vs. SFT qualitative comparison