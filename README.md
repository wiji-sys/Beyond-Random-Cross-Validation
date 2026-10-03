# Knowbase reproducibility package — revision working release

Manuscript: Beyond Random Cross-Validation: Evaluating Cross-Regional Generalization in Higher-Education Graduate Count Prediction
Package assembled: 3 October 2026

## Contents and execution
- original/pt_education_research_project/: original input data, notebooks 00–06, executed notebooks where available, and stored outputs. Follow that project's README and requirements to run the original workflow in numerical order.
- revision/Knowbase_Repeated_GroupCV/: 20 repeated province-disjoint five-fold evaluations. Follow its README and run run_repeated_groupcv.py.
- revision/Knowbase_Duan_Smearing/: correction sensitivity on the original random and grouped splits. Follow its README and run run_smearing.py.
- MANIFEST_SHA256.csv: paths, sizes, and SHA-256 checksums for included files.

## Source provenance
Original dataset URLs, release metadata, and source hashes are recorded in the Sumber_Data worksheet of original/pt_education_research_project/data/processed/Dataset_Gabungan_Kemdiktisaintek_2025.xlsx. The project retains 34 source-defined province categories, including legacy Papua and Papua Barat assignments. See the revised manuscript for geographic scope.

## Interpretation and versioning
The original results are preserved as the original analysis record. Statements there identifying Tweedie as best refer to the original fixed grouped partition, not all alternative partitions. The additional grouped analysis found Extra Trees and Poisson each ranked first in 9/20 repetitions, Tweedie in 1/20, and histogram gradient boosting in 1/20. Original Notebook 06 does not incorporate these revision analyses. Use the revision outputs together with the original outputs when assessing the revised manuscript.

Smearing uses fitted training residuals only. It was evaluated on the original random and grouped folds, not across the 20 additional allocations. Revision packages record their runtime versions and original-partition numerical replication checks. The original notebook workflow has not been rerun end-to-end in preparing this combined archive.

## Publication status
This is a prepared local archive, not a published repository release. No public repository URL or DOI has been assigned here. Before resubmission, deposit the author-approved package in a repository, check access from a signed-out session, and insert that real URL/DOI in the manuscript and response letter. Third-party source data retain their original terms; this package does not assign a new license to them.
