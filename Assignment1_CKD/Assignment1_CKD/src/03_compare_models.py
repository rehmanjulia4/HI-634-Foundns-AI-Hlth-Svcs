"""Stage 3: compare a baseline with three alternatives, then test the winner ONCE.

Run from the project root after stage 2:  python src/03_compare_models.py

Model choices use only the 320 training records (repeated stratified 5-fold CV).
The 80 test records are used one time, for the single selected model.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import (CATEGORICAL, FIG_DIR, NUMERIC, ORDINAL, PREDICTORS, PROC_DIR,
                    RANDOM_STATE, TAB_DIR, TARGET, wilson_ci)

NUM_COLS = NUMERIC + ORDINAL
df = pd.read_csv(PROC_DIR / "ckd_clean.csv")
split = pd.read_csv(PROC_DIR / "split_assignment.csv")
df = df.merge(split, on="record_id")
train, test = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
assert len(train) == 320 and len(test) == 80
Xtr, Xte = train[PREDICTORS], test[PREDICTORS]
ytr, yte = (train[TARGET] == "ckd").astype(int), (test[TARGET] == "ckd").astype(int)   # positive class = ckd
print(f"Train n={len(train)} (ckd {int(ytr.sum())}); test n={len(test)} (ckd {int(yte.sum())}). Positive class = ckd.")


# ---------------------------------------------------------------------------
def make_prep(strategy: str = "median_mode"):
    """Imputation strategies. All are fitted inside the CV folds, never on test data."""
    ind = strategy == "median_mode_indicator"
    cat_imp = (SimpleImputer(strategy="constant", fill_value="Unknown") if strategy == "unknown_category"
               else SimpleImputer(strategy="most_frequent"))
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=ind)),
                          ("sc", StandardScaler())]), NUM_COLS),
        ("cat", Pipeline([("imp", cat_imp),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])


def models():
    return {
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE),
        "Random forest (300 trees)": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                            random_state=RANDOM_STATE, n_jobs=1),
    }


COMPLEXITY = {"Logistic regression": 1, "Decision tree (depth 4)": 2, "Random forest (300 trees)": 3,
              "Majority-class baseline": 0}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
SCORING = {"recall": "recall", "precision": "precision", "f1": "f1", "macro_f1": "f1_macro", "accuracy": "accuracy"}


def cv_table(pipes: dict, X, y):
    rows = []
    for name, pipe in pipes.items():
        s = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=1, error_score="raise")
        r = {"model": name}
        for k in SCORING:
            r[f"{k}_mean"] = s[f"test_{k}"].mean()
            r[f"{k}_sd"] = s[f"test_{k}"].std()
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
print("\n" + "=" * 78, "\nA. MISSING-DATA TREATMENTS (training CV only)\n", "=" * 78, sep="")
complete_train = int(Xtr.notna().all(axis=1).sum())
print(f"Complete-case option would keep only {complete_train} of {len(Xtr)} training records "
      f"({100 * complete_train / len(Xtr):.1f}%) and would discard a different mix of classes:")
cc = train.loc[Xtr.notna().all(axis=1), TARGET].value_counts()
print(cc.to_string())
strategies = {"median_mode_indicator": "Median / most-frequent + numeric missing-value flags",
              "median_mode": "Median / most-frequent (no flags)",
              "unknown_category": "Median / 'Unknown' category"}
rows = []
for skey, slabel in strategies.items():
    for mname in ("Logistic regression", "Random forest (300 trees)"):
        t = cv_table({mname: Pipeline([("prep", make_prep(skey)), ("model", models()[mname])])}, Xtr, ytr)
        t.insert(0, "strategy", slabel)
        rows.append(t)
strat = pd.concat(rows, ignore_index=True)
show = strat[["strategy", "model", "recall_mean", "precision_mean", "macro_f1_mean", "accuracy_mean"]].round(4)
print(show.to_string(index=False))
strat.to_csv(TAB_DIR / "03_missing_data_strategies_cv.csv", index=False)

print("\n" + "=" * 78, "\nB. MODEL COMPARISON on training data (5-fold x 3 repeats, median/most-frequent imputation)\n",
      "=" * 78, sep="")
pipes = {n: Pipeline([("prep", make_prep("median_mode")), ("model", m)]) for n, m in models().items()}
comp = cv_table(pipes, Xtr, ytr)
print(comp[["model", "recall_mean", "recall_sd", "precision_mean", "f1_mean", "macro_f1_mean", "macro_f1_sd",
            "accuracy_mean"]].round(4).to_string(index=False))
comp.to_csv(TAB_DIR / "03_model_comparison_cv.csv", index=False)

print("\nWhy not select on recall alone: the majority-class baseline predicts 'ckd' for everyone, so its CKD recall is")
print(f"{comp.loc[0, 'recall_mean']:.2f} even though it never identifies a single notckd record. The rule below")
print("uses macro F1 as the primary measure and CKD recall as a guardrail.")
GUARD = 0.95
cand = comp[(comp.model != "Majority-class baseline") & (comp.recall_mean >= GUARD)].copy()
best = cand.macro_f1_mean.max()
near = cand[cand.macro_f1_mean >= best - 0.01].copy()
near["cx"] = near.model.map(COMPLEXITY)
selected = near.sort_values(["cx", "macro_f1_mean"], ascending=[True, False]).iloc[0].model
print(f"Rule: CV CKD recall >= {GUARD}; highest macro F1; if within 0.01 of the best, choose the simpler model.")
print(f"Models within 0.01 macro F1 of the best: {list(near.model)}")
print(f"SELECTED (using training data only): {selected}")

print("\n" + "=" * 78, "\nC. SINGLE LOOK AT THE TEST SET\n", "=" * 78, sep="")
final = Pipeline([("prep", make_prep("median_mode")), ("model", models()[selected])]).fit(Xtr, ytr)
pred = final.predict(Xte)
proba = final.predict_proba(Xte)[:, 1]
pd.DataFrame({'record_id': test.record_id, 'true_ckd': yte,
              'predicted_ckd': pred, 'probability_ckd': proba}).to_csv(
                  TAB_DIR / '03_test_predictions.csv', index=False)
cm = confusion_matrix(yte, pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix (rows = true, columns = predicted; order notckd, ckd):\n", cm)
print(f"True negatives {tn}, false positives {fp}, false negatives {fn}, true positives {tp}")
print(classification_report(yte, pred, target_names=["notckd", "ckd"], digits=3))
rng = np.random.default_rng(RANDOM_STATE)
boot = {"recall_ckd": [], "precision_ckd": [], "macro_f1": [], "accuracy": []}
for _ in range(2000):
    idx = rng.integers(0, len(yte), len(yte))
    yt, yp = yte.values[idx], pred[idx]
    boot["recall_ckd"].append(recall_score(yt, yp, zero_division=0))
    boot["precision_ckd"].append(precision_score(yt, yp, zero_division=0))
    boot["macro_f1"].append(f1_score(yt, yp, average="macro", zero_division=0))
    boot["accuracy"].append(accuracy_score(yt, yp))
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
rec_ci = wilson_ci(int(tp), int(tp + fn))
spec_ci = wilson_ci(int(tn), int(tn + fp))
test_metrics = {
    "selected_model": selected, "n_test": int(len(yte)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "accuracy": float(accuracy_score(yte, pred)),
    "recall_ckd": float(recall_score(yte, pred)), "precision_ckd": float(precision_score(yte, pred)),
    "f1_ckd": float(f1_score(yte, pred)),
    "specificity_notckd": float(tn / (tn + fp)),
    "macro_f1": float(f1_score(yte, pred, average="macro")),
    "bootstrap_95ci": ci, "bootstrap_note": "A perfect prediction vector gives degenerate percentile intervals; these do not establish certainty. Use the Wilson intervals and sampling limitations.", "wilson_95ci_recall": list(rec_ci), "wilson_95ci_specificity": list(spec_ci),
    "baseline_accuracy_from_class_share": float(yte.mean()),
}
print(json.dumps(test_metrics, indent=2))
pd.DataFrame(classification_report(yte, pred, target_names=["notckd", "ckd"], output_dict=True)).T \
    .round(4).to_csv(TAB_DIR / "03_test_classification_report.csv")

# Error analysis
errs = test.loc[pred != yte.values, ["record_id", TARGET, "hemo", "pcv", "sg", "al", "sc", "rbcc"]].copy()
errs["predicted"] = np.where(pred[pred != yte.values] == 1, "ckd", "notckd")
errs["prob_ckd"] = proba[pred != yte.values].round(3)
errs["n_missing_predictors"] = Xte.loc[pred != yte.values].isna().sum(axis=1).values
print("\nTest-set errors:")
print(errs.to_string(index=False) if len(errs) else "none: no test record was misclassified")
errs.to_csv(TAB_DIR / "03_test_errors.csv", index=False)
# Out-of-fold errors on the training set give more cases to study than the tiny test set
oof = cross_val_predict(Pipeline([("prep", make_prep("median_mode")), ("model", models()[selected])]),
                        Xtr, ytr, cv=5, method="predict")
oof_err = train.loc[oof != ytr.values, ["record_id", TARGET, "hemo", "pcv", "sg", "al", "sc", "rbcc"]].copy()
oof_err["predicted"] = np.where(oof[oof != ytr.values] == 1, "ckd", "notckd")
oof_err["n_missing_predictors"] = Xtr.loc[oof != ytr.values].isna().sum(axis=1).values
print(f"\nOut-of-fold training errors (5-fold, selected model): {len(oof_err)} of {len(train)}")
print(oof_err.to_string(index=False) if len(oof_err) else "none")
oof_err.to_csv(TAB_DIR / "03_training_oof_errors.csv", index=False)
test_metrics["oof_train_errors"] = int(len(oof_err))

print("\n" + "=" * 78, "\nD. SENSITIVITY CHECKS (training CV only)\n", "=" * 78, sep="")
sens = []
# D1: can the pattern of missingness alone predict the label?
miss_only = Xtr.isna().astype(int)
t = cv_table({"Missingness flags only (24 yes/no flags), logistic regression":
              Pipeline([("m", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])}, miss_only, ytr)
t.insert(0, "check", "D1 missingness pattern only")
sens.append(t)
# D2: remove the five strongest separators
drop5 = [r["variable"] for r in json.loads((TAB_DIR / "02_prepare_log.json").read_text())["strongest_separators"][:5]]
keep_num = [c for c in NUM_COLS if c not in drop5]
keep_cat = [c for c in CATEGORICAL if c not in drop5]


def prep_subset(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])


t = cv_table({f"{selected} without {', '.join(drop5)}":
              Pipeline([("prep", prep_subset(keep_num, keep_cat)), ("model", models()[selected])])}, Xtr, ytr)
t.insert(0, "check", "D2 drop five strongest separators")
sens.append(t)
sens = pd.concat(sens, ignore_index=True)
print(sens[["check", "model", "recall_mean", "precision_mean", "macro_f1_mean", "accuracy_mean"]].round(4).to_string(index=False))
sens.to_csv(TAB_DIR / "03_sensitivity_checks_cv.csv", index=False)
test_metrics["dropped_for_D2"] = drop5

print("\n" + "=" * 78, "\nE. WHICH FIELDS DRIVE THE MODELS (fitted on training data; association, not cause)\n", "=" * 78, sep="")
lr = Pipeline([("prep", make_prep("median_mode")), ("model", models()["Logistic regression"])]).fit(Xtr, ytr)
rf = Pipeline([("prep", make_prep("median_mode")), ("model", models()["Random forest (300 trees)"])]).fit(Xtr, ytr)
names = [n.split("__", 1)[1] for n in lr.named_steps["prep"].get_feature_names_out()]
coef = pd.Series(lr.named_steps["model"].coef_[0], index=names)
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
top_c = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10)
top_i = imp.sort_values(ascending=False).head(10)
print("Logistic regression, largest standardized coefficients (positive = toward ckd):\n", top_c.round(3).to_string())
print("\nRandom forest, largest impurity importances:\n", top_i.round(3).to_string())
pd.DataFrame({"lr_coef": coef, "rf_importance": imp}).sort_values("rf_importance", ascending=False) \
    .round(4).to_csv(TAB_DIR / "03_feature_importance.csv")

# ----------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
order = list(comp.model)
for ax, (m, lab) in zip(axes, [("macro_f1", "Macro F1"), ("recall", "CKD recall")]):
    ax.barh(order[::-1], comp[f"{m}_mean"].values[::-1], xerr=comp[f"{m}_sd"].values[::-1],
            color=["#4C72B0", "#4C72B0", "#4C72B0", "#999999"], capsize=3)
    for y, v in enumerate(comp[f"{m}_mean"].values[::-1]):
        ax.text(0.02, y, f"{v:.3f}", va="center", color="white", fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(f"{lab} (mean +/- SD over 15 validation folds)")
axes[1].set_yticklabels([])
fig.suptitle(f"Figure 5. Cross-validated comparison on the training set (n = {len(train)} records)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_cv_comparison.png", dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 4.6))
ConfusionMatrixDisplay(cm, display_labels=["notckd", "ckd"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"Figure 6. Test-set confusion matrix\n{selected} (n = {len(yte)} held-out records)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_confusion_matrix.png", dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
top_c[::-1].plot.barh(ax=axes[0], color=["#C44E52" if v > 0 else "#55A868" for v in top_c[::-1]])
axes[0].set_xlabel("Standardized coefficient (positive = toward ckd)")
axes[0].set_title("Logistic regression", fontsize=10)
top_i[::-1].plot.barh(ax=axes[1], color="#4C72B0")
axes[1].set_xlabel("Impurity importance")
axes[1].set_title("Random forest", fontsize=10)
fig.suptitle("Figure 7. Fields most associated with the label in training data (association, not cause)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig7_feature_importance.png", dpi=170)
plt.close(fig)

(TAB_DIR / "03_results.json").write_text(json.dumps(
    {"test": test_metrics, "complete_case_train": complete_train,
     "complete_case_class": {k: int(v) for k, v in cc.items()}}, indent=2))
print("\nSaved tables, Figures 5-7, and reports/tables/03_results.json.")
