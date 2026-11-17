# Independent ranking audit

PASS: all360 weighted confusion matrices and scores reproduced with sklearn. Maximum numeric difference: 2.78e-15. All24 scheme rankings agree.

| Class | Original contribution | Equal-class contribution | Shift contribution |
|---|---:|---:|---:|
|lymphoid|-0.02353|-0.00110|+0.02243|
|intestinal_metaplasia|+0.00106|+0.00516|+0.00410|
|tumor|+0.00488|+0.00844|+0.00356|
|mucosa_inflammation|-0.00119|+0.00151|+0.00270|
|other|-0.00309|-0.00358|-0.00049|

Contributions are (adaptive F1 minus shuffled F1)/5, averaged over three seeds. They add exactly to the macro-F1 contrast. Pure class reweighting preserves every class recall and balanced accuracy; it changes precision through the relative mass of false-positive source classes.

All24 weightings are reported in all_24_weightings.csv, including unfavorable results. Three seeds and24 scenarios are not72 independent cohorts. These are hypothetical reweightings of a validation set already used for stopping, not new test results or retraining interventions.
