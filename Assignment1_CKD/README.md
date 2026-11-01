# HI 634 Assignment 1 — Chronic Kidney Disease

Julia Rehman · University of Alabama at Birmingham · Dr. Rahul Sharma

## Question and source

How well do the 24 recorded measurements distinguish the supplied ckd and notckd labels, and how should missing measurements affect preparation? Each row is one clinical record. CKD is the positive class. This is classification of existing labels, not prediction of future disease.

Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic kidney disease [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020

Dataset and codebook: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

Course repository: https://github.com/rehmanjulia4/HI-634-Foundns-AI-Hlth-Svcs

Original supplied downloads date from October 2026; the exact original day is unknown. A fresh official archive was downloaded on November 1, 2026 and saved as data/raw/ckd/uci_source_download_2026-11-01.zip. Original files remain unchanged. The source codebook is chronic_kidney_disease.info.txt. No private patient data are used.

## Files

- reports/HI634_CKD_Report.docx: final APA student report, six main-text pages plus title page, references and supporting appendices.
- reports/HI634_CKD_Presentation.pptx: eight-slide presentation with no speaker notes.
- src/: the repeatable analysis, report builders and independent checker.
- data/raw/: unchanged public source files; included in the project ZIP, excluded from GitHub.
- data/processed/: cleaned table, fixed split membership and cached UCI importer copy.
- reports/figures/ and reports/tables/: labeled figures and detailed results.
- docs/DECISION_LOG.md: findings, preparation decisions and reasons.
- reports/run_log.txt, execution_manifest.json and independent_verification.json: completed-run evidence and verification.
- requirements.txt and requirements-lock.txt: packages and the exact tested Windows package versions.

## Run in Anaconda Prompt on Windows

Extract the project ZIP. Open Anaconda Prompt and change to the project folder containing this README. Close the Word report before rebuilding it.

For the first setup only:

```text
conda create --prefix ./.venv python=3.12 pip
conda activate ./.venv
python -m pip install -r requirements-lock.txt
```

For every run, from the project folder:

```text
conda activate ./.venv
python -c "import sys; print(sys.executable)"
python src/run_all.py
python src/verify_results.py
```

The Python path should point inside this project's .venv. The full run ends with SUCCESS. The checker ends with "all_assertions_passed": true. The run takes several minutes and rebuilds the audit, cleaned table, split, comparisons, figures and Word report. The PowerPoint is a separate final presentation of these results; it is not rebuilt by the analysis runner.

To browse the project for the demonstration, run `python -m jupyter lab` from that same activated environment. A notebook is not required; this project uses Python scripts.

The tested setup is Windows, Python 3.12.14. The lock file includes Windows-specific packages. The course packages are pandas, numpy, matplotlib, seaborn, scikit-learn and jupyter. The additional packages support UCI import and Office-file creation or inspection. requirements.txt is the portable package list; the lock file is the tested Windows environment. The manifest's in_virtual_environment field detects Python venvs and may be false for Conda; the recorded Python path identifies the separate Conda environment.

## Reproducing from GitHub

The complete project ZIP runs offline with its included raw files. A GitHub clone excludes data/raw and .venv. Download the UCI archive from the dataset page and extract the nested Chronic_Kidney_Disease.rar with a compatible archive tool. Place chronic_kidney_disease_full.arff and chronic_kidney_disease.info.txt in data/raw/ckd and retain the downloaded archives there. Then follow the run commands above. The saved importer copy is a cached source cross-check, not a fresh network retrieval. `python src/run_all.py --with-importer` optionally refreshes it; a network failure is reported.

## Verified results and limits

All 400 records remain: 250 ckd and 150 notckd. There are 1,012 missing predictor cells out of 9,600 and 158 complete records. A stratified split with seed 634 gives 320 training records (200 ckd, 120 notckd) and 80 test records (50 ckd, 30 notckd).

Training-only repeated five-fold cross-validation compares a majority baseline, logistic regression, a decision tree and a random forest. Mean macro F1 is .385, .993, .961 and .991 respectively. The selection rule uses CKD recall at least .95 and macro F1, preferring simplicity within .01 of the best score. Logistic regression correctly classifies all 80 test records: precision, recall and F1 are 1.000. A separate five-fold training check finds one error. Recall's approximate 95% Wilson interval is .929–1.000; specificity's is .886–1.000.

Preprocessing learns only from training folds. Original descriptive exploration used all 400 labels, so the test sample was not entirely unseen throughout the project. Label construction, measurement timing, representativeness and repeated-patient status remain uncertain. These results do not establish clinical readiness.
