"""Run the full evaluation suite and save a JSON + Markdown report.

Usage:
    .venv/bin/python scripts/run_eval.py [--skip-personalization]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# load .env
envf = ROOT / ".env"
if envf.exists():
    import os
    for line in envf.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from core.ingest import KnowledgeBase
from core.retrieve import Retriever
from core.evaluate import run_all


def main() -> None:
    kb = KnowledgeBase.load(ROOT / "data" / "kb.json")
    retriever = Retriever(kb)
    print(f"knowledge base: {len(kb.units)} units", flush=True)

    t0 = time.time()
    report = run_all(
        kb,
        retriever,
        ROOT / "tests" / "testset.json",
        skip_personalization="--skip-personalization" in sys.argv,
    )
    report["elapsed_sec"] = round(time.time() - t0, 1)
    report["kb_units"] = len(kb.units)

    out = ROOT / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nsaved to", out)


if __name__ == "__main__":
    main()
