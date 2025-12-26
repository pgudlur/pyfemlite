from __future__ import annotations
import argparse
import runpy
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(prog="pyfemlite", description="Minimal FEM solver demos.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run an example script (Python file path).")
    run.add_argument("path", type=str, help="Path to example (e.g., examples/poisson_mms.py)")

    args = parser.parse_args()

    if args.cmd == "run":
        p = Path(args.path)
        if not p.exists():
            raise FileNotFoundError(f"Example not found: {p}")
        runpy.run_path(str(p), run_name="__main__")
