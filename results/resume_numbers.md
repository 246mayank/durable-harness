# Numbers for the resume bullets

Generated with results.md; copy numbers from here only.

- N (seeded scenarios per config): **500**
- harness duplicate side effects: **0** (excess spend $0)
- X, retrying agent loop (`naive_retry`) duplicate side effects: **2898** across 423 runs (excess spend $133,400)
- Y, idempotency keys alone (`keys_only`) duplicate side effects: **1052** across 319 runs
  - of which on *keyed* endpoints (campaigns + ads), caused by a re-planned step producing new arguments: **369**
  - of which on endpoints with no key support (email batches + allocations): **683**
- harness escalations to a human: **228** in 184 runs; with conditional writes: **0**

Wording rules: say "simulated platform", "seeded scenarios", "built and evaluated", "scripted planner" and "in-process crash injection". Never "exactly-once", "production" or "deployed".
