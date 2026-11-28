"""
Backward-compatible single-series preprocessing entrypoint for Pune Tomato.
Delegates directly to the canonical leak-free pipeline in scripts/build_features.py
to guarantee identical lag definitions, rolling windows, and chronological splitting.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_features import build_model_ready_dataset, save_dataset


def main():
    df = build_model_ready_dataset(commodity="Tomato", market="Pune")
    out_path = save_dataset(df, "Tomato", "Pune")
    print(f"Leak-free Pune Tomato dataset saved to: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
