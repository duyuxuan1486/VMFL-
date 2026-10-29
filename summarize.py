#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("inputs", nargs="+", type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
rows = []
for path in args.inputs:
    rows.extend(json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
by_level = defaultdict(list)
for row in rows:
    by_level[int(row["level"])].append(row)
result = {
    "n": len(rows),
    "accuracy": sum(row["success"] for row in rows) / len(rows),
    "parse_failures": sum(row["parse_failure"] is not None for row in rows),
    "by_level": {
        str(level): {
            "n": len(level_rows),
            "correct": sum(row["success"] for row in level_rows),
            "accuracy": sum(row["success"] for row in level_rows) / len(level_rows),
        }
        for level, level_rows in sorted(by_level.items())
    },
}
args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
