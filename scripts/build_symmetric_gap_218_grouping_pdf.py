#!/usr/bin/env python3
"""Build the grouping-first Chinese proof of the 2.18 lower bound."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "thesis/symmetric_gap_218_grouping_zh.tex"
BUILD = ROOT / "tmp/pdfs/symmetric-gap-218-grouping"
OUTPUT = ROOT / "output/pdf/symmetric_gap_218_grouping_zh.pdf"


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    command = ["/Library/TeX/texbin/xelatex", "-interaction=nonstopmode",
               "-halt-on-error", "-file-line-error", f"-output-directory={BUILD}",
               str(SOURCE)]
    for run in range(1, 4):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (BUILD / f"compile-{run}.txt").write_text(result.stdout + result.stderr)
        if result.returncode:
            print(result.stdout[-7000:])
            raise SystemExit(result.returncode)
    shutil.copyfile(BUILD / (SOURCE.stem + ".pdf"), OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
