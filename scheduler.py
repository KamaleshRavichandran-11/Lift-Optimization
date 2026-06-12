"""
scheduler.py
Assigns elevators to active hall calls and recommends idle parking actions.
"""

from __future__ import annotations

from config import (
    ENERGY_IDLE_SEC,
    MAX_CAPACITY,
    PARKING_FLOORS,
    W1_DISTANCE,
    W2_STOPS,
    W3_LOAD,
    W4_DIRECTION,
    W5_IDLE,
)
from contracts import HallCall, LiftState


class SchedulerModule:
    def __init__(self, num_elevators: int, total_floors: int) -> None:
        self.num_elevators = num_elevators
        self.total_floors = total_floors

    @staticmethod
    def _direction_penalty(elevator_state: LiftState, floor: int) -> float:
        if elevator_state.direction == "IDLE":
            return 0.0
        if elevator_state.direction == "UP" and floor >= elevator_state.current_floor:
            return 0.0
        if elevator_state.direction == "DOWN" and floor <= elevator_state.current_floor:
            return 0.0
        return 2.0

    def calculate_assignment_cost(self, elevator_state: LiftState, hall_call: HallCall) -> float:
        distance = abs(elevator_state.current_floor - hall_call.floor_id)
        stops = len(elevator_state.scheduled_stops)
        load_factor = elevator_state.load / max(MAX_CAPACITY, 1)
        direction_penalty = self._direction_penalty(elevator_state, hall_call.floor_id)
        idle_penalty = 0.0 if elevator_state.direction == "IDLE" else 1.0
        duplicate_stop_bonus = -2.0 if hall_call.floor_id in elevator_state.scheduled_stops else 0.0

        return (
            (W1_DISTANCE * distance)
            + (W2_STOPS * stops)
            + (W3_LOAD * load_factor)
            + (W4_DIRECTION * direction_penalty)
            + (W5_IDLE * idle_penalty)
            + duplicate_stop_bonus
        )

    def assign_elevator(
        self,
        elevators_state: dict[str, LiftState],
        hall_call: HallCall,
    ) -> str | None:
        best_elevator: str | None = None
        min_cost = float("inf")

        for lift_id, state in elevators_state.items():
            if state.load >= MAX_CAPACITY:
                continue

            if hall_call.priority == "EMERGENCY":
                cost = float(abs(state.current_floor - hall_call.floor_id))
            else:
                cost = self.calculate_assignment_cost(state, hall_call)

            if cost < min_cost:
                min_cost = cost
                best_elevator = lift_id

        return best_elevator

    def recommend_parking_actions(
        self,
        elevators_state: dict[str, LiftState],
        has_active_calls: bool,
        seconds_since_demand: float,
    ) -> dict[str, int]:
        if has_active_calls or seconds_since_demand < ENERGY_IDLE_SEC:
            return {}

        actions: dict[str, int] = {}
        ordered_lifts = sorted(elevators_state.keys())

        for index, lift_id in enumerate(ordered_lifts):
            state = elevators_state[lift_id]
            if state.scheduled_stops or state.status == "DOORS_OPEN":
                continue

            parking_floor = PARKING_FLOORS[index % len(PARKING_FLOORS)]
            if state.current_floor != parking_floor:
                actions[lift_id] = parking_floor

        return actions
