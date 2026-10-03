# Knowbase Duan smearing sensitivity

Date: 2 October 2026. Input: the same 2,567 strata as original Notebook 03.

Run `python run_smearing.py` after installing requirements.txt. Input files are located relative to the script. model_definitions.py contains model/preprocessing functions from Notebook 03. No original project files are modified.

Random Forest and Extra Trees use the original fixed configurations and seed 42. The original five-fold random and province-disjoint assignments are reused. For each training fold, let z=log(1+Y), fitted log-count prediction zhat, and e=z-zhat. A global factor S=mean(exp(e)) is estimated using fitted TRAINING residuals only. Corrected predictions equal S*exp(zhat_test)-1, clipped below at 1e-9. This is not S*expm1(zhat_test). There is no test-outcome information in factor estimation. All preprocessing is fitted within the training fold.

The factor uses in-sample training residuals, not inner out-of-fold residuals. This is a classical global smearing sensitivity, not a guarantee of unbiased conditional means, especially with flexible tree fits or heterogeneous residual distributions. No tuning of smearing factors is performed.

All metrics in smearing_province_metrics.csv pool held-out stratum predictions, sum within each of 34 source-defined provinces, and evaluate the provincial totals. Uncorrected predictions reproduce stored original metrics to tolerance 1e-10 (see replication_audit.csv).

Grouped validation: RF RMSLE 0.247614 to 0.240150, MAE 11819.489 to 10363.537, total ratio 0.880812 to 0.913924. Extra Trees RMSLE 0.238169 to 0.234767, MAE 9260.136 to 8651.423, ratio 0.904854 to 0.924518.
Random validation: RF RMSLE 0.216075 to 0.206234; Extra Trees 0.218353 to 0.214517.

Smearing reduces, but does not eliminate, aggregate underprediction. These results concern the ORIGINAL two five-fold partitions. The 20-repeat grouping sensitivity is a separate experiment and was not rerun with corrected predictions. The original primary benchmark and repeated-CV ranking remain explicitly uncorrected log-count configurations.

Files: smearing_province_metrics.csv, training_smearing_factors.csv, oof_predictions.csv, replication_audit.csv, original_metrics.csv, original_fold_assignments.csv, input dataset, model code, requirements, and runtime versions.
