"""Shared helpers for the HI 634 CKD project.

Every stage script imports this module so that paths, column definitions and
the raw-file parser are defined exactly once.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 634
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "ckd"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"
for _p in (RAW_DIR, PROC_DIR, FIG_DIR, TAB_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Column definitions, taken from the UCI documentation (info.txt / ARFF header)
# ---------------------------------------------------------------------------
NUMERIC = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
# Documented as "nominal" by UCI, but the codes are ordered scales
# (specific gravity 1.005 to 1.025; albumin and sugar 0 to 5 on a dipstick).
ORDINAL = ["sg", "al", "su"]
CATEGORICAL = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
PREDICTORS = NUMERIC + ORDINAL + CATEGORICAL          # 24 predictors
TARGET = "class"
# The info.txt file and some tools spell two columns differently.
ALIASES = {"wc": "wbcc", "rc": "rbcc"}

UNITS = {
    "age": "years", "bp": "mm Hg", "bgr": "mg/dL", "bu": "mg/dL", "sc": "mg/dL",
    "sod": "mEq/L", "pot": "mEq/L", "hemo": "g/dL (UCI says 'gms')", "pcv": "% (volume)",
    "wbcc": "cells/cumm", "rbcc": "millions/cmm", "sg": "ordinal 1.005-1.025",
    "al": "ordinal 0-5", "su": "ordinal 0-5",
}
FULL_NAMES = {
    "age": "Age", "bp": "Blood pressure", "sg": "Specific gravity", "al": "Albumin",
    "su": "Sugar", "rbc": "Red blood cells", "pc": "Pus cell", "pcc": "Pus cell clumps",
    "ba": "Bacteria", "bgr": "Blood glucose random", "bu": "Blood urea",
    "sc": "Serum creatinine", "sod": "Sodium", "pot": "Potassium", "hemo": "Hemoglobin",
    "pcv": "Packed cell volume", "wbcc": "White blood cell count",
    "rbcc": "Red blood cell count", "htn": "Hypertension", "dm": "Diabetes mellitus",
    "cad": "Coronary artery disease", "appet": "Appetite", "pe": "Pedal edema",
    "ane": "Anemia",
}
MISSING_MARKERS = {"?", "", "nan", "NaN", "None"}


# ---------------------------------------------------------------------------
# Locating the raw file
# ---------------------------------------------------------------------------
def _try_extract(archive: Path, out_dir: Path) -> bool:
    """Try to unpack a .rar with whichever tool exists on this computer."""
    for tool, args in (
        ("7z", ["x", "-y", f"-o{out_dir}", str(archive)]),
        ("7zz", ["x", "-y", f"-o{out_dir}", str(archive)]),
        ("unrar", ["x", "-o+", str(archive), str(out_dir) + "/"]),
        ("bsdtar", ["-xf", str(archive), "-C", str(out_dir)]),
        ("tar", ["-xf", str(archive), "-C", str(out_dir)]),  # Windows 10/11 tar is bsdtar
    ):
        if shutil.which(tool):
            r = subprocess.run([tool, *args], capture_output=True, text=True)
            if r.returncode == 0 and list(out_dir.rglob("*.arff")):
                return True
    return False


def find_arff() -> Path:
    """Return the path of chronic_kidney_disease_full.arff inside data/raw/ckd.

    If only the UCI ZIP (which contains a RAR) is present, try to unpack it
    into data/raw/ckd/extracted. The original ZIP is never modified.
    """
    hits = sorted(RAW_DIR.rglob("*full*.arff"))
    if hits:
        return hits[0]
    out_dir = RAW_DIR / "extracted"
    out_dir.mkdir(exist_ok=True)
    for z in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out_dir)
    for rar in list(out_dir.rglob("*.rar")):
        _try_extract(rar, out_dir)
    hits = sorted(out_dir.rglob("*full*.arff"))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        "Could not find chronic_kidney_disease_full.arff. Open the UCI ZIP, extract "
        "Chronic_Kidney_Disease.rar with 7-Zip or WinRAR, and copy "
        "chronic_kidney_disease_full.arff into data/raw/ckd/ (do not edit it)."
    )


# ---------------------------------------------------------------------------
# Raw parser: reads every cell as text, changes nothing except the delimiter split
# ---------------------------------------------------------------------------
def parse_raw_arff(path: Path | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Read the ARFF data block as text.

    Returns (raw_df, anomalies). raw_df keeps whitespace and '?' exactly as
    written. A standard ARFF reader fails on this file because three rows do
    not have 25 comma-separated fields; those rows are repaired minimally and
    each repair is reported in `anomalies`.
    """
    path = path or find_arff()
    lines = path.read_text(encoding="latin-1").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "@data")
    names = [
        ln.split()[1].strip("'\"")
        for ln in lines
        if ln.lower().startswith("@attribute")
    ]
    names = [ALIASES.get(n, n) for n in names]
    assert len(names) == 25, f"expected 25 attributes, found {len(names)}"
    rows, anomalies = [], []
    for k, ln in enumerate(lines[start + 1:]):
        if not ln.strip():
            continue
        parts = ln.split(",")
        if len(parts) != 25:
            before = len(parts)
            if parts[-1].strip() == "":                      # trailing comma
                parts = parts[:-1]
                why = "removed trailing empty field"
            else:                                            # stray empty interior field
                idx = next(i for i, p in enumerate(parts) if p.strip() == "")
                parts = parts[:idx] + parts[idx + 1:]
                why = f"removed stray empty field at position {idx + 1}"
            anomalies.append({"data_row": len(rows) + 1, "fields_found": before,
                              "repair": why, "fields_after": len(parts)})
        assert len(parts) == 25, f"row {k} still has {len(parts)} fields"
        rows.append(parts)
    return pd.DataFrame(rows, columns=names, dtype=str), anomalies


# ---------------------------------------------------------------------------
# Cleaning (pure function so every stage uses the identical rules)
# ---------------------------------------------------------------------------
def clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, standardise '?' to NaN, convert numerics, map the label.

    No value is imputed here and no row is removed here.
    """
    df = raw.copy()
    for c in df.columns:
        df[c] = df[c].map(
            lambda v: np.nan if (pd.isna(v) or str(v).strip() in MISSING_MARKERS) else str(v).strip()
        )
    for c in NUMERIC + ORDINAL:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = df[TARGET].str.lower()
    df.insert(0, "record_id", np.arange(1, len(df) + 1))   # for tracing only, never a predictor
    return df


def load_clean() -> pd.DataFrame:
    raw, _ = parse_raw_arff()
    return clean_frame(raw)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (float(c - h), float(c + h))
