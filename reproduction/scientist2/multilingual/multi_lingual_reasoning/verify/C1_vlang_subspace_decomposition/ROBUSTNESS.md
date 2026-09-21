# Robustness Report — Claim C1

**Claim**: C1 — The hidden representations of Qwen-3-4B-Thinking contain a low-dimensional language-specific subspace V_lang that can be identified via SVD on mean-difference vectors, is reliably decodable by a linear classifier (language-identification accuracy > 90% on held-out languages), and is approximately orthogonal to the language-agnostic reasoning residual (principal angle cosine < 0.1).

**Verdict**: INTEGRITY_ONLY

**stage2_skip_reason**: max_verify_claims_cap

**robustness**: — (not computed — Stage 2 skipped per MAX_VERIFY_CLAIMS=1 cap)

**Baseline verdict**: not-supported (heldout_lang_acc=0.968 > 90% predicate MET, but complement_acc=0.491 >> 0.1 threshold fails the approximate-orthogonality predicate; claim predicate requires ALL three criteria)

**Phase 2 combined audit**: PASS (Exp=PASS, Mech=N/A)

**Why skipped**: Stage-1 audit passed (combined=PASS), but C3 was selected as the higher-importance claim for the single Stage-2 slot (MAX_VERIFY_CLAIMS=1). C1 is in the admitted pool and was deferred per `stage2_deferred_reason` in STAGE2_PICK.json.

**To upgrade**: `/auto-verify C1 -- resume: true` — Stage 1 audit (already PASS) will be reused; Stage 2 model-swap will run.

**Audit artifacts**: `verify/C1_vlang_subspace_decomposition/main_experiment_audit/`
