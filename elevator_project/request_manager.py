"""
request_manager.py
Tracks active hall calls, suppresses duplicates, and expires stale demand.
"""

from __future__ import annotations

import time

from config import CALL_CANCEL_EMPTY_SEC, CALL_REFRESH_SEC, SERVICE_COOLDOWN_SEC
from contracts import FloorDemand, HallCall, RequestUpdate


class RequestManager:
    def __init__(
        self,
        cancel_after_sec: float = CALL_CANCEL_EMPTY_SEC,
        refresh_interval_sec: float = CALL_REFRESH_SEC,
        service_cooldown_sec: float = SERVICE_COOLDOWN_SEC,
    ) -> None:
        self.cancel_after_sec = cancel_after_sec
        self.refresh_interval_sec = refresh_interval_sec
        self.service_cooldown_sec = service_cooldown_sec
        self._active_calls: dict[int, HallCall] = {}
        self._cooldowns: dict[int, float] = {}

    def update_from_demand(self, demand: FloorDemand, now: float | None = None) -> RequestUpdate:
        current_time = demand.timestamp if now is None else now
        update = RequestUpdate()
        existing = self._active_calls.get(demand.floor_id)

        cooldown_started_at = self._cooldowns.get(demand.floor_id)
        if cooldown_started_at is not None:
            if current_time - cooldown_started_at < self.service_cooldown_sec:
                return update
            del self._cooldowns[demand.floor_id]

        if not demand.demand_active:
            return update

        people_count = demand.waiting_count + demand.approaching_count
        if existing is None:
            call = HallCall(
                floor_id=demand.floor_id,
                priority=demand.priority,
                created_at=current_time,
                last_seen_at=current_time,
                people_count=people_count,
                crowd_score=demand.crowd_score,
            )
            self._active_calls[demand.floor_id] = call
            update.created.append(call)
            return update

        if current_time - existing.last_seen_at >= self.refresh_interval_sec:
            existing.last_seen_at = current_time
        else:
            existing.last_seen_at = current_time

        existing.priority = demand.priority
        existing.people_count = max(existing.people_count, people_count)
        existing.crowd_score = max(existing.crowd_score, demand.crowd_score)
        update.refreshed.append(existing)
        return update

    def expire_stale(self, now: float) -> list[HallCall]:
        expired: list[HallCall] = []

        for floor_id, call in list(self._active_calls.items()):
            if now - call.last_seen_at >= self.cancel_after_sec:
                call.status = "EXPIRED"
                expired.append(call)
                del self._active_calls[floor_id]

        return expired

    def get_assignable_calls(self) -> list[HallCall]:
        return [
            call
            for call in sorted(self._active_calls.values(), key=lambda item: item.floor_id)
            if call.status == "PENDING" and call.assigned_lift_id is None
        ]

    def mark_assigned(self, floor_id: int, lift_id: str) -> HallCall | None:
        call = self._active_calls.get(floor_id)
        if call is None:
            return None

        call.assigned_lift_id = lift_id
        call.status = "ASSIGNED"
        return call

    def mark_served(self, floor_id: int, served_at: float | None = None) -> HallCall | None:
        call = self._active_calls.pop(floor_id, None)
        if call is None:
            self._cooldowns[floor_id] = time.time() if served_at is None else served_at
            return None

        call.status = "SERVED"
        self._cooldowns[floor_id] = time.time() if served_at is None else served_at
        return call

    def active_calls(self) -> list[HallCall]:
        return sorted(self._active_calls.values(), key=lambda item: item.floor_id)

    def has_active_calls(self) -> bool:
        return bool(self._active_calls)
