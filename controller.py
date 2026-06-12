"""
controller.py
Simulates elevator movement, door dwell, service completion, and parking.
"""

from __future__ import annotations

import time

from config import DOOR_OPEN_TICKS
from contracts import LiftState, ServiceEvent


class ElevatorController:
    def __init__(self, lift_id: str, initial_floor: int = 1, parking_floor: int = 1) -> None:
        self.lift_id = lift_id
        self._door_hold_ticks = DOOR_OPEN_TICKS
        self.state = LiftState(
            lift_id=lift_id,
            current_floor=initial_floor,
            parking_floor=parking_floor,
        )

    def _sort_stops(self) -> None:
        current_floor = self.state.current_floor
        unique_stops = sorted(set(self.state.scheduled_stops))

        if self.state.direction == "UP":
            above = [floor for floor in unique_stops if floor >= current_floor]
            below = sorted((floor for floor in unique_stops if floor < current_floor), reverse=True)
            self.state.scheduled_stops = above + below
            return

        if self.state.direction == "DOWN":
            below = sorted((floor for floor in unique_stops if floor <= current_floor), reverse=True)
            above = [floor for floor in unique_stops if floor > current_floor]
            self.state.scheduled_stops = below + above
            return

        self.state.scheduled_stops = sorted(unique_stops, key=lambda floor: (abs(floor - current_floor), floor))

    def add_stop(self, floor: int) -> None:
        if floor not in self.state.scheduled_stops:
            self.state.scheduled_stops.append(floor)
            self._sort_stops()

    def has_stop(self, floor: int) -> bool:
        return floor in self.state.scheduled_stops

    def cancel_stop(self, floor: int) -> None:
        if floor in self.state.scheduled_stops:
            self.state.scheduled_stops = [stop for stop in self.state.scheduled_stops if stop != floor]
            self._sort_stops()

    def park_at(self, floor: int) -> None:
        self.state.parking_floor = floor
        if not self.has_stop(floor) and self.state.current_floor != floor:
            self.add_stop(floor)

    def move(self) -> ServiceEvent | None:
        if self.state.status == "DOORS_OPEN" and self.state.door_open_ticks > 0:
            self.state.door_open_ticks -= 1
            if self.state.door_open_ticks == 0:
                self.state.status = "STOPPED"
                if not self.state.scheduled_stops:
                    self.state.direction = "IDLE"
                    if self.state.current_floor == self.state.parking_floor:
                        self.state.status = "PARKED"
            return None

        if not self.state.scheduled_stops:
            self.state.target_floor = None
            self.state.direction = "IDLE"
            self.state.idle_ticks += 1
            self.state.status = "PARKED" if self.state.current_floor == self.state.parking_floor else "STOPPED"
            return None

        self.state.idle_ticks = 0
        self._sort_stops()
        next_stop = self.state.scheduled_stops[0]
        self.state.target_floor = next_stop

        if self.state.current_floor < next_stop:
            self.state.current_floor += 1
            self.state.direction = "UP"
            self.state.status = "MOVING"
            return None

        if self.state.current_floor > next_stop:
            self.state.current_floor -= 1
            self.state.direction = "DOWN"
            self.state.status = "MOVING"
            return None

        self.state.scheduled_stops.pop(0)
        self.state.status = "DOORS_OPEN"
        self.state.direction = "IDLE"
        self.state.door_open_ticks = self._door_hold_ticks
        return ServiceEvent(
            lift_id=self.lift_id,
            floor_id=next_stop,
            event_type="served_floor",
            timestamp=time.time(),
        )

    def get_state(self) -> LiftState:
        return LiftState(
            lift_id=self.state.lift_id,
            current_floor=self.state.current_floor,
            direction=self.state.direction,
            status=self.state.status,
            load=self.state.load,
            scheduled_stops=list(self.state.scheduled_stops),
            target_floor=self.state.target_floor,
            door_open_ticks=self.state.door_open_ticks,
            idle_ticks=self.state.idle_ticks,
            parking_floor=self.state.parking_floor,
        )
