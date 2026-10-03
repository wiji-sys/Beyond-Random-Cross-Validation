# Knowbase: repeated province-disjoint CV sensitivity analysis

Analysis date: 2 October 2026.

## Design
Same 2,567 strata and 34 source-defined province categories as Notebook 03. Seven model configurations and preprocessing functions copied from Notebook 03. Model random seed remains 42. No hyperparameter search and no smearing correction in this analysis.

Twenty repetitions were fixed before inspecting results, using NumPy default_rng seeds 4201–4220. Each repetition randomly permutes the sorted 34 province labels and divides them into five test groups of 7,7,7,7,6 provinces. This balances group counts, not stratum counts or outcomes. Each province is held out exactly once per repetition. No source province overlaps training and test data. All seven models use identical splits within each repetition. All preprocessing and exposure-weight normalization are fitted on the training fold only.

For each model and repetition, held-out stratum predictions are pooled and summed within each province. RMSLE is computed across the 34 province totals, not averaged across fold RMSLEs. Rank uses this pooled province RMSLE within each repetition. SD is the sample standard deviation across repetitions and is descriptive split sensitivity, not a confidence interval. Repetitions reuse the same observations and must not be treated as independent samples for significance tests.

## Reproduction
Python 3.12. Install requirements.txt; run `python run_repeated_groupcv.py` from any working directory. The supplied dataset is read relative to the script. Four worker processes are used. Versions are in versions.json. For reproducibility and limited CPU use, set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 before launching.

The original nonshuffled GroupKFold partition was also rerun with these functions in the present environment. All seven province RMSLE values reproduced the original stored results to machine precision (maximum absolute difference 5.6e-17). See original_partition_replication.csv. The current interpreter differs from the original Python 3.13.5 execution; this audit checks the original benchmark under the current environment.

## Results
Extra Trees: mean RMSLE 0.238227, SD 0.007367; first in 9/20 repetitions.
Poisson: mean 0.238592, SD 0.008776; first in 9/20.
Tweedie: mean 0.242059, SD 0.009270; first in 1/20.
Histogram gradient boosting: first in 1/20.
Tweedie minus Poisson paired RMSLE: mean +0.003467, range -0.001397 to +0.007088. Tweedie has lower RMSLE in 1/20.

The original Tweedie first rank is not stable across allocations. This sensitivity analysis does not establish universal superiority of Extra Trees or Poisson. Original partition results remain valid for that partition.

## Files
ranking_summary.csv: all-model summary for supplementary reporting.
repeat_metrics.csv: per-repetition metrics and ranks.
province_fold_assignments.csv: all group assignments (680 rows).
province_predictions.csv: observed/predicted provincial totals.
oof_repeat_*.csv: held-out predictions for each stratum and model.
02_modeling_dataset.csv: input dataset.
run_repeated_groupcv.py: standalone runnable analysis.
original_partition_replication.csv: numerical replication audit.
