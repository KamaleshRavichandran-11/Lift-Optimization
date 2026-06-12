"""
decision.py
Converts tracked people into elevator demand signals.
"""

from __future__ import annotations

import math

from config import (
    APPROACH_THRESHOLD,
    CROWD_THRESHOLD,
    THRESHOLD,
    W1_VELOCITY,
    W2_DIRECTION,
    W3_DISTANCE,
    W4_WAITING_ZONE,
    W5_DWELL,
    WAITING_DWELL_SEC,
)
from contracts import FloorDemand, PersonTrack, VisionSnapshot


class DecisionModule:
    def __init__(self) -> None:
        self._threshold = float(THRESHOLD)
        self._approach_threshold = float(APPROACH_THRESHOLD)

    @staticmethod
    def sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def predict_intent(self, track: PersonTrack) -> float:
        zone_bonus = 1.0 if track.in_waiting_zone else 0.4 if track.in_approach_zone else 0.0
        dwell_score = min(track.seen_for_sec / max(WAITING_DWELL_SEC, 0.1), 1.0)
        score = (
            (W1_VELOCITY * track.speed)
            + (W2_DIRECTION * math.cos(track.motion_angle))
            - (W3_DISTANCE * track.distance_m)
            + (W4_WAITING_ZONE * zone_bonus)
            + (W5_DWELL * dwell_score)
        )
        return self.sigmoid(score)

    def classify_track(self, track: PersonTrack) -> tuple[str, float]:
        intent_score = self.predict_intent(track)
        label = "PASSING"

        if (
            track.in_waiting_zone
            and track.seen_for_sec >= WAITING_DWELL_SEC
            and intent_score >= self._threshold * 0.85
        ):
            label = "WAITING"
        elif track.in_waiting_zone and track.speed <= 0.05 and track.seen_for_sec >= 0.75:
            label = "WAITING"
        elif track.in_approach_zone and intent_score >= self._approach_threshold:
            label = "APPROACHING"

        track.intent_score = intent_score
        track.intent_label = label
        return label, intent_score

    def analyze_snapshot(self, snapshot: VisionSnapshot, floor_id: int) -> FloorDemand:
        waiting_count = 0
        approaching_count = 0
        passing_count = 0
        active_track_ids: list[str] = []
        notes: list[str] = []

        for track in snapshot.people:
            label, score = self.classify_track(track)
            notes.append(f"{track.track_id}:{label}:{score:.2f}")

            if label == "WAITING":
                waiting_count += 1
                active_track_ids.append(track.track_id)
            elif label == "APPROACHING":
                approaching_count += 1
                active_track_ids.append(track.track_id)
            else:
                passing_count += 1

        crowd_score = waiting_count + (0.5 * approaching_count)
        priority = "HIGH" if crowd_score >= CROWD_THRESHOLD else "NORMAL"
        demand_active = waiting_count > 0 or approaching_count > 0

        return FloorDemand(
            floor_id=floor_id,
            timestamp=snapshot.timestamp,
            demand_active=demand_active,
            dispatch_ready=demand_active,
            priority=priority,
            waiting_count=waiting_count,
            approaching_count=approaching_count,
            passing_count=passing_count,
            crowd_score=crowd_score,
            active_track_ids=active_track_ids,
            notes=notes,
        )
