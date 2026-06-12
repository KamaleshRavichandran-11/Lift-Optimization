"""
analytics.py
Lightweight CSV logging for frame summaries and call lifecycle events.
"""

from __future__ import annotations

import csv
import json
import os
import time

from contracts import FloorDemand, HallCall, LiftState, VisionSnapshot


class AnalyticsLogger:
    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        self._handle = open(self.log_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=["timestamp", "kind", "frame", "floor", "details"],
        )
        if self._handle.tell() == 0:
            self._writer.writeheader()
            self._handle.flush()

    def _write(self, kind: str, frame: int | None, floor: int | None, details: dict[str, object]) -> None:
        self._writer.writerow(
            {
                "timestamp": f"{time.time():.3f}",
                "kind": kind,
                "frame": "" if frame is None else frame,
                "floor": "" if floor is None else floor,
                "details": json.dumps(details, sort_keys=True),
            }
        )
        self._handle.flush()

    def log_frame(
        self,
        frame_number: int,
        snapshot: VisionSnapshot,
        demand: FloorDemand,
        active_calls: list[HallCall],
        elevators: dict[str, LiftState],
    ) -> None:
        details = {
            "source": snapshot.source,
            "people_detected": snapshot.count,
            "waiting_count": demand.waiting_count,
            "approaching_count": demand.approaching_count,
            "passing_count": demand.passing_count,
            "priority": demand.priority,
            "active_calls": [call.floor_id for call in active_calls],
            "elevators": {
                lift_id: {
                    "floor": state.current_floor,
                    "status": state.status,
                    "direction": state.direction,
                    "stops": list(state.scheduled_stops),
                }
                for lift_id, state in elevators.items()
            },
        }
        self._write("frame", frame_number, demand.floor_id, details)

    def log_event(self, kind: str, details: dict[str, object]) -> None:
        floor = details.get("floor_id")
        self._write(kind, None, int(floor) if isinstance(floor, int) else None, details)

    def close(self) -> None:
        self._handle.close()
