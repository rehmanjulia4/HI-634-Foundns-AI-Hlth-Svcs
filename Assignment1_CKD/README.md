# HI 634 Assignment 1 — Chronic Kidney Disease

Julia Rehman · University of Alabama at Birmingham

## Project question

How well can the recorded clinical information distinguish records labeled chronic kidney disease (CKD) from those labeled non-CKD?

Each row is one clinical record. This project compares predictions with the dataset's existing labels. It does not establish early detection or readiness for healthcare use.

## Start here

- **Paper:** `reports/HI634_CKD_Report.docx`
- **Presentation:** `reports/HI634_CKD_Presentation.pptx`
- **Run the project:** `src/run_all.py`
- **Check the results independently:** `src/verify_results.py`

The project uses Python scripts; a separate notebook is not required. The presentation is reviewed separately and is not rebuilt by the analysis command.

The current presentation has nine slides and ends with **References**. The report is titled **Classifying Chronic Kidney Disease Records**.

## Folder guide

| Item | Contents |
|---|---|
| `data/raw/ckd/` | Original downloads and codebook. Cleaning never overwrites them. Included in the complete ZIP; excluded from GitHub. |
| `data/processed/` | Cleaned table, training/test assignments, and saved UCI importer copy. |
| `src/` | Data review, preparation, model comparisons, figure/report builders, and independent checker. |
| `reports/figures/` | Figures created by the analysis. The presentation shows the same verified results using editable charts and tables. |
| `reports/figures/apa/` | The three figures used in the paper. Other figures in `reports/figures/` support the analysis. |
| `reports/tables/` | Data-quality checks, comparisons, predictions, and other supporting results. |
| `docs/DECISION_LOG.md` | Findings, decisions, and reasons. |
| `docs/report_template.docx` | The accepted paper's wording and layout, used when rebuilding the report. This is not a second submission paper. |
| `docs/report_sources.json` | Evidence supporting the accepted paper. The builder checks this before recreating it. |
| `requirements.txt` | Required packages for Windows, Mac, Linux, or online Anaconda. |
| `requirements-lock.txt` | Exact versions from the tested local Windows environment. |
| `README.md` | Data-source and download instructions, environment setup, exact run commands, and expected results. |
| `.gitignore` | Excludes `.venv/`, `data/raw/`, and temporary files from Git tracking. |

The paper uses these three figures in `reports/figures/apa/`:

- `figure1_missingness.png`: incomplete records and the five fields most often missing.
- `figure2_class_balance.png`: 250 CKD and 150 non-CKD records.
- `figure3_confusion_matrix.png`: the 80 test predictions compared with existing labels.

The seven figures directly inside `reports/figures/` are supporting analysis outputs. Their numbering belongs to the analysis workflow and differs from the paper's figure numbers.

## Get the original data

Source: Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). *Chronic kidney disease* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020

Download page and codebook: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

The supplied downloads date from October 2026; the original day is unknown. A fresh official archive was downloaded on November 1, 2026. Both copies are preserved in the complete project ZIP.

**If using that ZIP:** extract it and keep its folders together. The original files are already included.

**If downloading from GitHub:** download the UCI archive above. Extract the ZIP, then its nested `Chronic_Kidney_Disease.rar` with a compatible archive tool. Create `data/raw/ckd/` and put these files there:

- `chronic_kidney_disease_full.arff`
- `chronic_kidney_disease.info.txt`

Keep the original downloaded archives there as well. Do not edit the original data. No private patient data are used.

## Set up Python once

Open a prompt or terminal **in the project folder containing this README**.

For local Anaconda:

```text
conda create --prefix ./.venv python=3.12 pip
conda activate ./.venv
python -m pip install -r requirements.txt
```

To use the exact tested Windows package versions, install `requirements-lock.txt` instead. That file contains Windows-specific packages and should not be used in online Anaconda, Mac, or Linux.

If not using Anaconda, create a Python 3.12 virtual environment, activate it, and install `requirements.txt`.

## Run and check

From that same project folder, with the environment active:

```text
python -c "import sys; print(sys.executable)"
python src/run_all.py
python src/verify_results.py
```

The first command identifies the Python being used. The full run should end with **SUCCESS**. The checker should end with **"all_assertions_passed": true**.

The full run reviews the original files, prepares the data, compares models, recreates the three report figures, and rebuilds the paper. Close the project's output Word file before rebuilding it. The accepted report template is kept separately and is not overwritten.

The report builder preserves the accepted wording, tables, citations, and page format. It checks the new results against the evidence supporting that paper. If results change at the reported precision, it stops and requests a review rather than creating a paper with unsupported numbers.

### Online Anaconda / JupyterLab

Upload and extract the complete project, keeping its folder structure. Open a Terminal from JupyterLab's Launcher, change into the project folder, and use the run/check commands above. The online service already provides a Python environment; a local Windows environment does not need to stay open.

If packages are missing, install them with `python -m pip install -r requirements.txt`. In a notebook opened inside the project folder, the equivalent commands are:

```python
%pip install -r requirements.txt
%run src/run_all.py
%run src/verify_results.py
```

Install packages only when needed. Restart the notebook kernel if installation requests it. `requirements-lock.txt` is for local Windows, not the online service.

## Expected results

- 400 records retained: 250 CKD and 150 non-CKD.
- 1,012 missing entries across the 24 input fields; 158 complete records.
- Training: 320 records, including 200 CKD and 120 non-CKD. Test: 80 records, including 50 CKD and 30 non-CKD. Split seed: 634.
- Mean training-comparison macro F1: baseline 0.385; logistic regression 0.993; decision tree 0.961; random forest 0.991.
- Selected model: logistic regression. It matched all 80 test labels; test precision, recall, and F1 were 1.000. A separate training check found one error.

`reports/run_log.txt` records the run. `reports/execution_manifest.json` records Python/package versions and stage completion. `reports/independent_verification.json` records the independent checks.

The analysis was previously run locally in Anaconda and in online Anaconda with matching reported results. Exact package versions are recorded in each run's manifest.

## Limits and source cross-check

The saved UCI importer copy is a cached cross-check, not a fresh download. To request a fresh importer retrieval, use `python src/run_all.py --with-importer`; a network failure is reported.

Model preparation and selection use training data. The original descriptive exploration included all 400 labels, so test records were not completely unseen throughout the project. Diagnosis procedures, measurement timing, repeat visits, and applicability to other populations remain uncertain. Perfect scores in this small test group do not establish clinical readiness.

Course repository: https://github.com/rehmanjulia4/HI-634-Foundns-AI-Hlth-Svcs
