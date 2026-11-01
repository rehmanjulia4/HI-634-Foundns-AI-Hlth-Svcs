"""Stage 1: inspect the raw UCI CKD archive BEFORE any cleaning.

Run from the project root:  python src/01_audit_raw.py [--with-importer]

Prints every finding so it can be read before it is interpreted. Writes tables
to reports/tables and two figures to reports/figures. Raw files are never edited.
"""
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CATEGORICAL, FIG_DIR, FULL_NAMES, NUMERIC, ORDINAL, PREDICTORS,
                    PROC_DIR, RAW_DIR, TAB_DIR, TARGET, clean_frame, find_arff,
                    parse_raw_arff)

ap = argparse.ArgumentParser()
ap.add_argument("--with-importer", action="store_true",
                help="also download via ucimlrepo (id=336) and compare missing counts")
args = ap.parse_args()

findings = {}
print("=" * 78, "\n1. FILE INVENTORY\n", "=" * 78, sep="")
inv = []
for p in sorted(RAW_DIR.rglob("*")):
    if p.is_file():
        inv.append({"file": str(p.relative_to(RAW_DIR)), "bytes": p.stat().st_size})
inv = pd.DataFrame(inv)
print(inv.to_string(index=False))
inv.to_csv(TAB_DIR / "01_file_inventory.csv", index=False)
arff_path = find_arff()
print("\nAnalysis file:", arff_path.name)

raw, anomalies = parse_raw_arff(arff_path)
print("\n" + "=" * 78, "\n2. STRUCTURE\n", "=" * 78, sep="")
print(f"Rows: {raw.shape[0]}   Columns: {raw.shape[1]} (24 predictors + class)")
print(f"Rows whose comma count did not match the header: {len(anomalies)}")
for a in anomalies:
    print("  ", a)
pd.DataFrame(anomalies).to_csv(TAB_DIR / "01_row_structure_anomalies.csv", index=False)
print("\nFirst 5 raw rows (text exactly as stored):")
print(raw.head(5).to_string())
findings["raw_rows"] = int(raw.shape[0])
findings["raw_cols"] = int(raw.shape[1])
findings["row_structure_anomalies"] = len(anomalies)

print("\n" + "=" * 78, "\n3. RAW MARKERS AND WHITESPACE (before cleaning)\n", "=" * 78, sep="")
q = (raw == "?").sum()
ws = raw.apply(lambda s: (s != s.str.strip()).sum())
print("Cells equal to '?':", int(q.sum()), " | cells with leading/trailing whitespace or tabs:", int(ws.sum()))
print("\nColumns with whitespace problems:")
print(ws[ws > 0].to_string())
print("\nRaw class spellings (repr shows hidden tabs/spaces):")
raw_class = raw[TARGET].map(repr).value_counts()
print(raw_class.to_string())
print("\nRaw spellings of other text columns that differ from their clean value:")
dirty = []
for c in CATEGORICAL:
    vc = raw[c].map(repr).value_counts()
    for k, v in vc.items():
        if k.strip("'") != k.strip("'").strip() or "\\t" in k:
            dirty.append({"column": c, "raw_value": k, "n": int(v)})
dirty = pd.DataFrame(dirty)
print(dirty.to_string(index=False) if len(dirty) else "none")
dirty.to_csv(TAB_DIR / "01_dirty_text_values.csv", index=False)
raw_class.rename_axis("raw_class_value").reset_index(name="n").to_csv(TAB_DIR / "01_raw_class_spellings.csv", index=False)
q_strip = (raw.apply(lambda s: s.str.strip()) == "?").sum()
print(f"Cells equal to '?' after trimming whitespace: {int(q_strip.sum())} "
      f"({int(q_strip.sum() - q.sum())} were hidden by a tab or space)")
findings["question_mark_cells"] = int(q.sum())
findings["question_mark_cells_after_trim"] = int(q_strip.sum())
findings["whitespace_cells"] = int(ws.sum())

df = clean_frame(raw)   # cleaning is shown here ONLY to measure; nothing is dropped or filled

print("\n" + "=" * 78, "\n4. TYPES: non-numeric text hiding in numeric columns\n", "=" * 78, sep="")
bad = {}
for c in NUMERIC + ORDINAL:
    s = raw[c].str.strip()
    s = s[~s.isin(["?", ""])]
    n_bad = int(pd.to_numeric(s, errors="coerce").isna().sum())
    if n_bad:
        bad[c] = n_bad
print("Values that are not '?' and cannot be read as numbers:", bad or "none")
print("(All columns arrive as text in a raw read because '?' appears in every numeric field family.)")

print("\n" + "=" * 78, "\n5. MISSINGNESS\n", "=" * 78, sep="")
miss = pd.DataFrame({
    "variable": PREDICTORS,
    "description": [FULL_NAMES[c] for c in PREDICTORS],
    "missing_n": [int(df[c].isna().sum()) for c in PREDICTORS],
})
miss["missing_pct"] = (100 * miss.missing_n / len(df)).round(1)
miss["missing_pct_in_ckd"] = [round(100 * df.loc[df[TARGET] == "ckd", c].isna().mean(), 1) for c in PREDICTORS]
miss["missing_pct_in_notckd"] = [round(100 * df.loc[df[TARGET] == "notckd", c].isna().mean(), 1) for c in PREDICTORS]
miss = miss.sort_values("missing_n", ascending=False).reset_index(drop=True)
print(miss.to_string(index=False))
miss.to_csv(TAB_DIR / "01_missingness_by_variable.csv", index=False)
per_row = df[PREDICTORS].isna().sum(axis=1)
complete = int((per_row == 0).sum())
print(f"\nTotal missing predictor cells: {int(df[PREDICTORS].isna().sum().sum())} of {len(df) * 24} "
      f"({100 * df[PREDICTORS].isna().sum().sum() / (len(df) * 24):.1f}%)")
print(f"Records with NO missing predictor (complete cases): {complete} of {len(df)}")
print("Missing predictors per record (distribution):")
print(per_row.describe().round(2).to_string())
by_class_missing = per_row.groupby(df[TARGET]).mean().round(2)
print("Mean missing predictors per record by class:\n", by_class_missing.to_string())
findings.update({
    "missing_cells": int(df[PREDICTORS].isna().sum().sum()),
    "total_cells": int(len(df) * 24),
    "complete_cases": complete,
    "most_missing_variable": miss.iloc[0].variable,
    "most_missing_n": int(miss.iloc[0].missing_n),
    "most_missing_pct": float(miss.iloc[0].missing_pct),
    "mean_missing_per_row_ckd": float(by_class_missing["ckd"]),
    "mean_missing_per_row_notckd": float(by_class_missing["notckd"]),
})

print("\n" + "=" * 78, "\n6. DUPLICATES\n", "=" * 78, sep="")
dups = int(df.drop(columns="record_id").duplicated().sum())
print("Fully duplicated records (after trimming whitespace):", dups)
findings["duplicate_rows"] = dups

print("\n" + "=" * 78, "\n7. CLASS BALANCE (after trimming the label)\n", "=" * 78, sep="")
cc = df[TARGET].value_counts()
print(cc.to_string(), "\nProportions:", (cc / len(df)).round(3).to_dict())
findings["class_counts"] = {k: int(v) for k, v in cc.items()}
findings["class_variants_raw"] = {k: int(v) for k, v in raw[TARGET].map(repr).value_counts().items()}

print("\n" + "=" * 78, "\n8. NUMERIC RANGES vs UCI DEFINITIONS (flag, do NOT delete)\n", "=" * 78, sep="")
rng = df[NUMERIC + ORDINAL].describe().T[["count", "min", "25%", "50%", "75%", "max"]].round(3)
print(rng.to_string())
rng.to_csv(TAB_DIR / "01_numeric_ranges.csv")
# Plausibility screens are broad adult-lab sanity limits chosen for flagging only.
screens = {"bp": (40, 200), "sod": (110, 160), "pot": (2, 8), "hemo": (3, 20), "pcv": (10, 60),
           "rbcc": (1.5, 8), "wbcc": (2000, 30000), "sc": (0.3, 20), "age": (1, 100)}
flag_rows = []
for c, (lo, hi) in screens.items():
    m = df[c].notna() & ((df[c] < lo) | (df[c] > hi))
    for rid, v in zip(df.loc[m, "record_id"], df.loc[m, c]):
        flag_rows.append({"record_id": int(rid), "variable": c, "value": float(v),
                          "screen": f"{lo}-{hi}", "class": df.loc[df.record_id == rid, TARGET].iloc[0]})
flags = pd.DataFrame(flag_rows)
print("\nValues outside broad plausibility screens (kept in the data, flagged only):")
print(flags.to_string(index=False) if len(flags) else "none")
flags.to_csv(TAB_DIR / "01_implausible_value_flags.csv", index=False)
findings["flagged_values"] = len(flags)

if args.with_importer:
    print("\n" + "=" * 78, "\n9. CROSS-CHECK WITH THE UCI PYTHON IMPORTER\n", "=" * 78, sep="")
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=336)
        imp = ds.data.features.copy()
        imp["class"] = ds.data.targets.iloc[:, 0]
        imp.to_csv(PROC_DIR / "uci_importer_copy.csv", index=False)   # working copy, not raw
        print("Importer shape:", imp.shape)
        print("Importer columns:", list(imp.columns))
        imp.columns = [c if c not in ("wc", "rc") else {"wc": "wbcc", "rc": "rbcc"}[c] for c in imp.columns]
        imp_miss = imp[PREDICTORS].isna().sum() + (imp[PREDICTORS].astype(str).apply(lambda s: s.str.strip()) == "?").sum()
        cmp_ = pd.DataFrame({"archive_missing": df[PREDICTORS].isna().sum(), "importer_missing": imp_miss})
        cmp_["difference"] = cmp_.importer_missing - cmp_.archive_missing
        print(cmp_.to_string())
        cmp_.to_csv(TAB_DIR / "01_archive_vs_importer_missing.csv")
        print("Importer class spellings:", imp["class"].map(repr).value_counts().to_dict())
    except Exception as exc:                                       # network or package problem
        print("Importer comparison skipped:", type(exc).__name__, exc)

# ---------------------------------------------------------------- figures
miss_plot = miss.sort_values("missing_pct")
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(miss_plot.variable, miss_plot.missing_pct, color="#4C72B0")
for y, v in enumerate(miss_plot.missing_pct):
    ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Records with a missing value (% of 400)")
ax.set_ylabel("Predictor")
ax.set_title("Figure 1. Missing values by predictor (n = 400 records)")
ax.set_xlim(0, miss_plot.missing_pct.max() + 8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_missingness.png", dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
raw_vc = raw[TARGET].map(repr).value_counts()
axes[0].bar(range(len(raw_vc)), raw_vc.values, color="#C44E52")
axes[0].set_xticks(range(len(raw_vc)))
axes[0].set_xticklabels(raw_vc.index, rotation=20, ha="right", fontsize=8)
for i, v in enumerate(raw_vc.values):
    axes[0].text(i, v + 3, str(v), ha="center")
axes[0].set_title("Raw class spellings (repr shows hidden characters)")
axes[0].set_ylabel("Records")
axes[1].bar(["notckd", "ckd"], [cc["notckd"], cc["ckd"]], color=["#55A868", "#C44E52"])
for i, v in enumerate([cc["notckd"], cc["ckd"]]):
    axes[1].text(i, v + 3, f"{v} ({100 * v / len(df):.1f}%)", ha="center")
axes[1].set_title("After trimming whitespace")
axes[1].set_ylabel("Records")
fig.suptitle("Figure 2. Class balance (n = 400 records)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_class_balance.png", dpi=170)
plt.close(fig)

(TAB_DIR / "01_findings.json").write_text(json.dumps(findings, indent=2))
print("\nSaved tables to reports/tables and Figures 1-2 to reports/figures.")
print(json.dumps(findings, indent=2))
