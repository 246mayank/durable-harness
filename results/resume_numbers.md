# Numbers for the resume bullets

Generated with results.md; copy numbers from here only.

- N (seeded scenarios per config): **500**
- harness duplicate side effects: **0** (excess spend $0); 95% upper bound on the per-run duplicate rate: **0.60%**
- X, retrying agent loop (`naive_retry`) duplicate side effects: **3467** across 452 runs (excess spend $185,000)
- Y, idempotency keys alone (`keys_only`) duplicate side effects: **1253** across 367 runs
  - of which on *keyed* endpoints (campaigns + ads), caused by a re-planned step producing new arguments: **366**
  - of which on endpoints with no key support (email batches + allocations): **887**
- harness escalations to a human: **333** in 232 runs; with conditional writes: **0**
- delayed delivery: harness without deadlines: **34** duplicate effects in 31 runs; with deadlines: **0**
- zombie incarnations (200 scenarios, 189 zombies): no fencing or deadlines **529** duplicate effects; fencing only **27**; both **0**
- parallel tool calls (same 500 scenarios, independent calls decided at once): harness duplicate side effects **0**, `keys_only` **1038**; planner calls per run 5.0 instead of 9.0

Wording rules: say "simulated platform", "seeded scenarios" and "built and evaluated". The numbers above come from the scripted planner with in-process crash injection; the hard-kill numbers (separate OS processes) are in sigkill.md, the fault-rate sweep in sweep.md, every crash schedule in exhaustive.md, the journal's cost in overhead.md, and the real-LLM runs in llm.md and llm_local.md, so cite those files for those claims. Never "exactly-once", "production" or "deployed".
