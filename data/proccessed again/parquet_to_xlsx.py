"""
parquet_to_xlsx.py
==================
Convert a parquet file in this directory tree to .xlsx for inspection.

By default converts processed/ml_ready_dataset.parquet, but accepts any
parquet path as a positional argument.

Run:
    # default — converts ml_ready_dataset.parquet
    python "data/proccessed again/parquet_to_xlsx.py"

    # specific file
    python "data/proccessed again/parquet_to_xlsx.py" \
        "data/proccessed again/processed/pair_features.parquet"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SRC = SCRIPT_DIR / "processed" / "ml_ready_dataset.parquet"


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.exists():
        print(f"ERROR: not found: {src}", file=sys.stderr)
        sys.exit(1)

    out = src.with_suffix(".xlsx")
    print(f"[load]  {src}")
    df = pd.read_parquet(src)
    print(f"        shape: {df.shape}")

    if len(df) > 1_048_575:
        print(f"[warn]  {len(df):,} rows exceeds Excel's 1,048,576 limit; will truncate")
        df = df.head(1_048_575)

    print(f"[write] {out}")
    try:
        df.to_excel(out, index=False, engine="openpyxl")
    except ImportError:
        print("ERROR: openpyxl not installed. Run: pip install openpyxl", file=sys.stderr)
        sys.exit(2)
    print("[done]")


if __name__ == "__main__":
    main()
