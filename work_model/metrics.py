"""Small metric logger for training and evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class MetricLogger:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []

    def record(self, row: dict[str, Any]) -> None:
        self.records.append(dict(row))

    def write_csv(self, filename: str = "metrics.csv") -> Path:
        path = self.run_dir / filename
        if not self.records:
            return path

        fieldnames: list[str] = []
        for row in self.records:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        return path

    def write_json(self, payload: dict[str, Any], filename: str = "summary.json") -> Path:
        path = self.run_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        return path

