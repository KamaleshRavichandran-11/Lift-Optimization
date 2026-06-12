"""
contracts.py
Shared data models used across the smart lift MVP.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PersonTrack:
    track_id: str
    x_norm: float
    y_norm: float
    bbox_height: float
    distance_m: float
    speed: float
    motion_angle: float
    in_waiting_zone: bool
    in_approach_zone: bool
    seen_for_sec: float
    last_seen_ts: float
    intent_score: float = 0.0
    intent_label: str = "UNKNOWN"


@dataclass
class VisionSnapshot:
    timestamp: float
    people: list[PersonTrack] = field(default_factory=list)
    source: str = "simulation"

    @property
    def count(self) -> int:
        return len(self.people)

    def summary(self) -> dict[str, object]:
        return {
            "source": self.source,
            "count": self.count,
            "track_ids": [track.track_id for track in self.people],
        }


@dataclass
class FloorDemand:
    floor_id: int
    timestamp: float
    demand_active: bool
    dispatch_ready: bool
    priority: str
    waiting_count: int
    approaching_count: int
    passing_count: int
    crowd_score: float
    active_track_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class HallCall:
    floor_id: int
    priority: str
    created_at: float
    last_seen_at: float
    people_count: int = 0
    crowd_score: float = 0.0
    status: str = "PENDING"
    assigned_lift_id: str | None = None

    def age(self, now: float) -> float:
        return max(0.0, now - self.created_at)


@dataclass
class RequestUpdate:
    created: list[HallCall] = field(default_factory=list)
    refreshed: list[HallCall] = field(default_factory=list)
    expired: list[HallCall] = field(default_factory=list)


@dataclass
class LiftState:
    lift_id: str
    current_floor: int
    direction: str = "IDLE"
    status: str = "STOPPED"
    load: int = 0
    scheduled_stops: list[int] = field(default_factory=list)
    target_floor: int | None = None
    door_open_ticks: int = 0
    idle_ticks: int = 0
    parking_floor: int = 1


@dataclass
class ServiceEvent:
    lift_id: str
    floor_id: int
    event_type: str
    timestamp: float
