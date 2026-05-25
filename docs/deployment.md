# Case Study: Encrypted Webshell Detection

This case study illustrates VAPD's production capability on one of the most adversarial NTD scenarios: **encrypted webshell communication**. It expands on Section 5.2 of the paper, which references this case as evidence that VAPD-derived custom-sized models can match the accuracy of the 7B teacher on hard, real-world traffic.

## Background

A **webshell** is a malicious script uploaded to a compromised server that allows the attacker to execute arbitrary commands remotely. Once installed, the attacker interacts with the shell over what looks like normal HTTP traffic. **Encrypted webshells** add an additional layer of evasion: the command and response payloads are encrypted (often AES- or XOR-based) before transmission, so the traffic on the wire contains no plaintext attack signatures.

This combination breaks signature-based IDS:

- **No plaintext payload** to match regex patterns against
- **Statistically indistinguishable** from benign encrypted traffic at the byte level
- **Polymorphic** — the same shell produces different ciphertexts on each interaction

Detecting encrypted webshells therefore requires **semantic-level reasoning** over the HTTP transaction structure, not just payload inspection: anomalous request/response patterns, unusual content-length distributions, suspicious endpoint behavior, and contextual signals across the full transaction.

## Production Detection Example

The screenshot below is taken from the Sangfor Athena XDR console, showing a real production detection by the **VAPD-derived 4.5B variant** on encrypted webshell communication.

<p align="center">
  <img src="figures/webshell_detection.png" width="750">
  <br>
  <em>Real-world detection of encrypted Webshell communication by the VAPD-4.5B model, deployed in Sangfor Athena XDR. (Sensitive customer information has been redacted.)</em>
</p>

## Why This Matters for VAPD

This case demonstrates three properties that distinguish VAPD's production behavior:

### 1. The 4.5B variant performs at teacher-level on the hardest cases

Standard 3B models (whether obtained via CPT+SFT or competing pruning baselines) tend to degrade specifically on encrypted/obfuscated traffic, where the absence of surface-level signatures means the model must rely entirely on deeper structural reasoning. The VAPD-4.5B variant, pruned directly from the 7B teacher under domain-adaptive importance estimation, **preserves this reasoning capacity** and detects encrypted webshells with accuracy fully aligned with the 7B baseline (see [`../docs/additional_results.md`](../docs/additional_results.md) for the full Mistral family scaling analysis).

### 2. Hardware-adaptive sizing pays off operationally

The 4.5B size is not arbitrary — it was chosen to fit the GPU memory budget of a specific class of Sangfor security appliances. Replacing the previous 7B baseline with this 4.5B variant on the same hardware yielded:

- **2.125× higher single-node throughput**
- **1.582× lower average HTTP inspection latency**

Without VAPD, this size point would have been unavailable: open-source Qwen-2.5 releases at 1.5B / 3B / 7B leave a gap between 3B (insufficient accuracy on encrypted webshells) and 7B (over-budget for the target appliance). VAPD bridges this gap.

### 3. Semantic reasoning generalizes beyond plaintext attacks

VAPD-derived models learn to recognize attack **structure**, not just **strings**. Even when the malicious payload itself is encrypted and unreadable, the surrounding transaction — request cadence, response size distribution, endpoint behavior, header anomalies — provides the contextual signals that the model can reason over. This is exactly the capability that signature-based IDS lacks (see [`../docs/domain_background.md §1`](../docs/domain_background.md#1-why-llms-for-network-traffic-detection)).

## Related Documents

- 🌐 [`../docs/domain_background.md`](../docs/domain_background.md) — Why LLMs are needed for NTD beyond signature matching
- 🧬 [`../docs/additional_results.md`](../docs/additional_results.md) — Mistral 4.5B variant analysis
- 🔍 [`vapd_vs_baseline.md`](vapd_vs_baseline.md) — VAPD vs. CPT+SFT vs. SFT on Command Injection