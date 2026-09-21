# Reviewer Memory

Persistent suspicion log across iterations (append-only). Reviewer: external `gpt-5.4` via the
documented HTTP fallback (MCP `llm-chat` tool not exposed this host; endpoint reachable, genuine review).

## Iteration 1 — Score: 7/10, Verdict: almost

- **New suspicions** (verbatim from reviewer `## Memory update`):
  - C1 appears genuinely supported: strong pre-registered monotone dose-response at site 28, winning-setting helix gain +0.068, specificity controls clean, verify-stage model-swap replication helpful.
  - Main caveat for C1: effect at the validity-preserving setting is statistically solid but modest; strongest gains occur where validity degrades.
  - Length shortening is a persistent concern: matched-band analysis reduces but does not fully eliminate reviewer suspicion that steering exploits ORF/length regime shifts.
  - Off-targets are substantial and CAA-specific, especially GC drop (−0.139) and protein length drop (~−37%); paper must not describe these as "not degraded."
  - C2's actual NI claim is fine and should survive review if stated narrowly as validity non-inferiority only.
  - Proxy-chain concern (ORF heuristic + ESMFold/DSSP endpoint) may limit how strongly biological structure claims can be stated.
  - Within-family checkpoint transfer is a good robustness point, but broader external validity is still limited.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**:
  - Proxy-heavy endpoint chain limits strength of biological structure claims (paper-framing item).
  - Broader external validity beyond within-family checkpoint swap (would need a cross-family model or wet-lab-linked validation).
  - C2 deferred swap-test (length-regressed endpoint / alternate pLDDT policy) not yet run — `stage2_skip_reason: max_verify_claims_cap`.
- **Patterns**:
  - Off-target coupling (length, GC) is intrinsic to the CAA lever and transfers across checkpoints — recurring across experiment + verify + review; must be framed honestly rather than dismissed. This is the single cross-cutting theme, and the C2 overclaim wording was its most concrete symptom (addressed this iteration via a type-⓪ paper-side tightening).
