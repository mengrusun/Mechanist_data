#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw both panels. Each is a separate process, so a failure in one panel
leaves the other untouched and reports its own traceback.

    python3 run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    HERE / "e_domain_heatmap" / "plot_e_domain_heatmap.py",
    HERE / "f_hypothesis_quality" / "plot_f_hypothesis_quality.py",
]


def main() -> int:
    failed = []
    for script in SCRIPTS:
        print(f"== {script.relative_to(HERE)}")
        if subprocess.run([sys.executable, str(script)], cwd=script.parent).returncode:
            failed.append(script.name)
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
