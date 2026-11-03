"""Stage 2: clean, split, and check for leakage. Run from the project root.

Order matters: the split happens BEFORE any imputer, encoder or scaler is fitted.
Nothing is imputed in the saved clean table; imputation lives inside the
scikit-learn Pipelines of stage 3.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import (CATEGORICAL, FIG_DIR, NUMERIC, ORDINAL, PREDICTORS, PROC_DIR,
                    RANDOM_STATE, TAB_DIR, TARGET, TEST_SIZE, clean_frame, parse_raw_arff)

raw, _ = parse_raw_arff()
log = {}

print("=" * 78, "\nA. CLEANING: before vs after\n", "=" * 78, sep="")
raw_missing_cells = int((raw[PREDICTORS].apply(lambda s: s.str.strip()) == "?").sum().sum())
df = clean_frame(raw)
print(f"Rows before: {len(raw)}   rows after: {len(df)}  (no row removed; no label was missing)")
print(f"Missing predictor cells: raw '?' after trimming = {raw_missing_cells}; "
      f"NaN after cleaning = {int(df[PREDICTORS].isna().sum().sum())}")
print("Class variants combined:",
      raw[TARGET].map(repr).value_counts().to_dict(), "->", df[TARGET].value_counts().to_dict())
assert df[TARGET].isin(["ckd", "notckd"]).all(), "unexpected class label"
assert len(df) == 400
dtypes = pd.DataFrame({"dtype_after_cleaning": df[PREDICTORS].dtypes.astype(str)})
print(dtypes.T.to_string())
df.to_csv(PROC_DIR / "ckd_clean.csv", index=False)
log.update(rows_before=len(raw), rows_after=len(df), missing_cells_raw=raw_missing_cells,
           missing_cells_clean=int(df[PREDICTORS].isna().sum().sum()))

print("\n" + "=" * 78, "\nB. STRATIFIED TRAIN/TEST SPLIT (80/20, seed 634)\n", "=" * 78, sep="")
train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET])
split_tab = pd.DataFrame({
    "all": df[TARGET].value_counts(), "train": train[TARGET].value_counts(), "test": test[TARGET].value_counts()
}).T[["ckd", "notckd"]]
split_tab["total"] = split_tab.sum(axis=1)
split_tab["ckd_share"] = (split_tab.ckd / split_tab.total).round(3)
print(split_tab.to_string())
assert set(train.record_id).isdisjoint(test.record_id)
split_tab.to_csv(TAB_DIR / "02_split_class_counts.csv")
df.assign(split=np.where(df.record_id.isin(test.record_id), "test", "train"))[["record_id", "split"]] \
    .to_csv(PROC_DIR / "split_assignment.csv", index=False)
log["train_n"], log["test_n"] = len(train), len(test)
log["train_class"] = train[TARGET].value_counts().to_dict()
log["test_class"] = test[TARGET].value_counts().to_dict()

print("\n" + "=" * 78, "\nC. LEAKAGE CHECKS\n", "=" * 78, sep="")
print("Predictors used:", len(PREDICTORS))
assert TARGET not in PREDICTORS and "record_id" not in PREDICTORS
print("1) 'class' and 'record_id' are not predictors: OK")
# Single-variable discrimination on TRAINING data only, to see whether any one field nearly reveals the answer.
y_tr = (train[TARGET] == "ckd").astype(int)
rows = []
for c in NUMERIC + ORDINAL:
    s = train[c]
    ok = s.notna()
    auc = roc_auc_score(y_tr[ok], s[ok])
    rows.append({"variable": c, "n_observed": int(ok.sum()), "auc_ckd": round(auc, 3),
                 "auc_strength": round(max(auc, 1 - auc), 3)})
for c in CATEGORICAL:
    s = train[c]
    ok = s.notna()
    # share of the majority class inside each category, weighted: a purity score
    pur = train[ok].groupby(c)[TARGET].agg(lambda g: g.value_counts().iloc[0]).sum() / ok.sum()
    rows.append({"variable": c, "n_observed": int(ok.sum()), "auc_ckd": np.nan,
                 "auc_strength": round(float(pur), 3)})
leak = pd.DataFrame(rows).sort_values("auc_strength", ascending=False)
print("2) Strongest single-variable separators in the training set (AUC for numeric fields,")
print("   category purity for categorical fields; 1.0 would mean the field alone reveals the label):")
print(leak.head(10).to_string(index=False))
leak.to_csv(TAB_DIR / "02_single_variable_separation_train.csv", index=False)
print("3) Timing: UCI gives no measurement dates, so we cannot prove every field was recorded before the label.")
print("   Several fields (hemoglobin, serum creatinine, albumin, specific gravity) are the kind of tests used to")
print("   diagnose CKD, so high accuracy may partly reflect how the label was defined. Document this limitation.")
log["strongest_separators"] = leak.head(6)[["variable", "auc_strength"]].to_dict("records")

print("\n" + "=" * 78, "\nD. IMPUTATION: distributions before vs after (fit on TRAIN only)\n", "=" * 78, sep="")
num_all = NUMERIC + ORDINAL
imp = SimpleImputer(strategy="median").fit(train[num_all])
after = pd.DataFrame(imp.transform(train[num_all]), columns=num_all, index=train.index)
ba = []
for c in num_all:
    n_miss = int(train[c].isna().sum())
    if n_miss:
        ba.append({"variable": c, "train_missing_n": n_miss,
                   "median_before": train[c].median(), "median_after": after[c].median(),
                   "mean_before": round(train[c].mean(), 3), "mean_after": round(after[c].mean(), 3),
                   "sd_before": round(train[c].std(), 3), "sd_after": round(after[c].std(), 3)})
ba = pd.DataFrame(ba).sort_values("train_missing_n", ascending=False)
ba["sd_change_pct"] = (100 * (ba.sd_after / ba.sd_before - 1)).round(1)
print(ba.to_string(index=False))
ba.to_csv(TAB_DIR / "02_imputation_before_after_train.csv", index=False)
log["max_sd_shrink_pct"] = float(ba.sd_change_pct.min())

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, c, lab in zip(axes, ["rbcc", "wbcc", "pot"], ["Red blood cell count (millions/cmm)",
                                                       "White blood cell count (cells/cumm)", "Potassium (mEq/L)"]):
    obs = train[c].dropna()
    if c == "pot":
        obs = obs[obs < 10]
        a = after[c][after[c] < 10]
    else:
        a = after[c]
    ax.hist(obs, bins=20, alpha=.6, label=f"observed (n={len(obs)})", color="#4C72B0")
    ax.hist(a, bins=20, alpha=.5, label=f"after median fill (n={len(a)})", color="#DD8452")
    ax.set_xlabel(lab, fontsize=8)
    ax.set_ylabel("Training records")
    ax.legend(fontsize=7)
fig.suptitle("Figure 3. Median imputation adds a spike at the median and narrows the spread (training set, n = 320;\n"
             "potassium panel excludes two extreme values for display only)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_imputation_before_after.png", dpi=170)
plt.close(fig)

# Exploration figure: relationship between key labs and class (all 400 records, exploration only)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
for ax, c, lab in zip(axes, ["hemo", "sc", "pcv", "sg"],
                      ["Hemoglobin (g/dL)", "Serum creatinine (mg/dL, log scale)", "Packed cell volume (%)",
                       "Specific gravity"]):
    data = [df.loc[df[TARGET] == k, c].dropna() for k in ("notckd", "ckd")]
    ax.boxplot(data, tick_labels=[f"notckd\n(n={len(data[0])})", f"ckd\n(n={len(data[1])})"])
    ax.set_ylabel(lab, fontsize=8)
    if c == "sc":
        ax.set_yscale("log")
fig.suptitle("Figure 4. Four laboratory fields by class, observed values only (exploration, all 400 records)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_key_labs_by_class.png", dpi=170)
plt.close(fig)

(TAB_DIR / "02_prepare_log.json").write_text(json.dumps(log, indent=2, default=float))
print("\nSaved: data/processed/ckd_clean.csv, split_assignment.csv; tables; Figures 3-4.")
