#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def get_epoch_summary(path: Path) -> dict:
    data = load_json(path)
    run = data.get("0") or data.get(0)
    if run is None:
        raise KeyError(f"Missing epoch 0 in {path}")
    return run


def extract_sequence_results(path: Path) -> list[int]:
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, dict):
        if "results" in data:
            return [int(result) for result in data["results"]]
        if "sequences" in data:
            return [int(item["success_count"]) for item in data["sequences"]]
    if isinstance(data, list):
        return [int(item["success_count"]) if isinstance(item, dict) else int(item) for item in data]
    raise TypeError(f"Unsupported sequence_results.json format in {path}")


def get_chain_sr(summary: dict, index: int) -> float:
    chain_sr = summary.get("chain_sr", {})
    return float(chain_sr.get(str(index), chain_sr.get(index, 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path)
    args = parser.parse_args()

    result_files = sorted(args.eval_root.glob("worker_*/results.json"))
    if not result_files:
        raise FileNotFoundError(f"No worker results found under {args.eval_root}/worker_*/results.json")

    results = []
    task_success = Counter()
    task_total = Counter()
    summaries = []
    for path in result_files:
        summary = get_epoch_summary(path)
        summaries.append(summary)
        for task, info in summary.get("task_info", {}).items():
            task_success[task] += int(info["success"])
            task_total[task] += int(info["total"])
        results.extend(extract_sequence_results(path.parent / "sequence_results.json"))

    if results:
        count = Counter(results)
        chain_sr = {
            str(i): sum(count[j] for j in reversed(range(i, 6))) / len(results)
            for i in range(1, 6)
        }
        avg_seq_len = float(np.mean(results))
        num_sequences = len(results)
    else:
        avg_seq_len = float(np.mean([summary["avg_seq_len"] for summary in summaries]))
        chain_sr = {
            str(i): float(np.mean([get_chain_sr(summary, i) for summary in summaries]))
            for i in range(1, 6)
        }
        num_sequences = None

    task_info = {
        task: {
            "success": int(task_success[task]),
            "total": int(task_total[task]),
            "sr": task_success[task] / task_total[task] if task_total[task] else 0.0,
        }
        for task in sorted(task_total)
    }
    merged = {
        "num_workers": len(result_files),
        "num_sequences": num_sequences,
        "avg_seq_len": avg_seq_len,
        "chain_sr": chain_sr,
        "task_info": task_info,
        "worker_result_files": [str(path) for path in result_files],
    }
    out = args.eval_root / "merged_results.json"
    out.write_text(json.dumps(merged, indent=2, sort_keys=True))
    print(json.dumps(merged, indent=2, sort_keys=True))
    print(f"merged results saved to {out}")


if __name__ == "__main__":
    main()
