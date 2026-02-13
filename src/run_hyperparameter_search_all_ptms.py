#!/usr/bin/env python3
"""
Run hyperparameter search for every PTM type that uses the standard dataset schema.

Outputs (results, best config, summary) are written under docs/ptm_<PTM_TYPE>/.
Requires: papermill (pip install papermill).

Usage (from project root or from src/):
  python src/run_hyperparameter_search_all_ptms.py
  python run_hyperparameter_search_all_ptms.py   # if run from src/
"""

import subprocess
import sys
from pathlib import Path

# PTM types with standard schema (original_sequence, ptm_type, UniProt_ID + numeric features).
# crot_k is skipped: it has a different schema (no embedding/entropy columns).
PTM_TYPES = ["acet_k", "gly_n", "phos_y", "sumo_k"]
MAX_CONFIGS = 150

def main():
    src_dir = Path(__file__).resolve().parent
    notebook = src_dir / "hyperparameter_search.ipynb"
    if not notebook.exists():
        print(f"Notebook not found: {notebook}")
        sys.exit(1)

    # Run from src/ so relative paths in the notebook (../datasets, ../docs) are correct
    for ptm in PTM_TYPES:
        out_dir = src_dir.parent / "docs" / f"ptm_{ptm}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_notebook = out_dir / "execution.ipynb"

        print(f"\n{'='*60}")
        print(f"Running hyperparameter search for PTM type: {ptm}")
        print(f"Output directory: {out_dir}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, "-m", "papermill",
            str(notebook),
            str(out_notebook),
            "-p", "PTM_TYPE", ptm,
            "-p", "MAX_CONFIGS", str(MAX_CONFIGS),
        ]
        ret = subprocess.run(cmd, cwd=str(src_dir))
        if ret.returncode != 0:
            print(f"Papermill failed for {ptm} (exit code {ret.returncode}).")
            sys.exit(ret.returncode)

    print("\nAll PTM types completed. Results are in docs/ptm_<PTM_TYPE>/.")


if __name__ == "__main__":
    main()
