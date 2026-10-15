"""Run the full evaluation and write eval_report.json (+ docs/EVALUATION.md generated from it).

Usage:
    python scripts/run_eval.py                  # everything; LLM-judged metrics run only if an LLM is reachable
    python scripts/run_eval.py --skip-generation
    python scripts/run_eval.py --write-gate     # also persist the calibrated refusal gate to core/gate.json

Every number in the report comes from this script; nothing is hand-edited.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import llm  # noqa: E402

llm.load_env()

from core import evaluate  # noqa: E402


def main() -> None:
    t0 = time.time()
    testset = ROOT / "tests" / "testset.json"
    # calibrate the refusal gate first so the generation stage already uses it
    cases = evaluate.load_testset(testset)
    courses = {n: evaluate.build_course(n) for n in evaluate.COURSES}
    gate = evaluate.calibrate_gate(courses, cases)
    os.environ["RETRIEVAL_MIN_SCORE"] = str(gate["min_score"])
    os.environ["RETRIEVAL_MIN_COVERAGE"] = str(gate["min_coverage"])
    os.environ["RETRIEVAL_MIN_SCORE_ONLINE"] = str(gate["min_score_online"])
    if "--write-gate" in sys.argv:
        (ROOT / "core" / "gate.json").write_text(
            json.dumps({"min_score": gate["min_score"], "min_coverage": gate["min_coverage"],
                        "min_score_online": gate["min_score_online"], "chosen_on": gate["chosen_on"]}, indent=1), encoding="utf-8")
        print("wrote core/gate.json:", gate["min_score"], gate["min_coverage"])

    report = evaluate.run_all(testset, skip_generation="--skip-generation" in sys.argv)
    report["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report["elapsed_sec"] = round(time.time() - t0, 1)
    out = ROOT / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("retrieval", "retrieval_gate", "assessment")}, indent=1)[:2500])
    print("saved", out)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_eval_doc.py")], check=False)


if __name__ == "__main__":
    main()
