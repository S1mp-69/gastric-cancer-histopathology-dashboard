# Stain-assignment feasibility: training-only

Sample:160 training patches,32/class, three stain draws each; probe alpha=.05, primary stronger alpha=.5. Initial and frozen trained-last checkpoints evaluated without fitting.

Trained checkpoint within-class rank correlation: 0.6881; descriptive patient-cluster interval [0.6171, 0.7489].
Cross-direction correlation: 0.2345.
Predefined engineering gate passed: True.

| Tissue | Same-direction Spearman |
|---|---:|
|tumor|0.5536|
|mucosa_inflammation|0.7241|
|intestinal_metaplasia|0.6069|
|lymphoid|0.7480|
|other|0.8079|

This measures loss-response prediction only. Even a passed gate does not establish classification improvement, beneficial assignment, true morphology preservation, novelty, or publication readiness.

No new validation/test pixels were accessed; the trained checkpoint historically used validation for early stopping. No automatic training campaign launched. The sample/thresholds are engineering choices, and probe/target share a stain direction by design. This can inflate apparent predictability relative to unseen directions. FP32 inference reduces mixed-precision rounding but uint8 reconstruction still quantizes small probes. Geometric/RGB jitter was omitted here; utility in the full training pipeline remains untested.

Next: if the gate passed, test the exact within-class assignment invariant and actual loss under ranked versus random strengths before any classifier comparison; otherwise investigate the failed surrogate without searching validation for a favorable replacement.
