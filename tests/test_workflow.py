from __future__ import annotations

import math
import pathlib
import sys
import unittest

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from contracts import FloorDemand, HallCall, LiftState, PersonTrack, VisionSnapshot
from decision import DecisionModule
from request_manager import RequestManager
from scheduler import SchedulerModule


class DecisionModuleTests(unittest.TestCase):
    def test_waiting_and_passing_classification(self) -> None:
        decision = DecisionModule()
        snapshot = VisionSnapshot(
            timestamp=100.0,
            people=[
                PersonTrack(
                    track_id="1",
                    x_norm=0.50,
                    y_norm=0.70,
                    bbox_height=320.0,
                    distance_m=1.2,
                    speed=0.01,
                    motion_angle=0.0,
                    in_waiting_zone=True,
                    in_approach_zone=False,
                    seen_for_sec=2.0,
                    last_seen_ts=100.0,
                ),
                PersonTrack(
                    track_id="2",
                    x_norm=0.10,
                    y_norm=0.35,
                    bbox_height=190.0,
                    distance_m=2.8,
                    speed=0.45,
                    motion_angle=math.pi,
                    in_waiting_zone=False,
                    in_approach_zone=False,
                    seen_for_sec=0.4,
                    last_seen_ts=100.0,
                ),
            ],
        )

        demand = decision.analyze_snapshot(snapshot, floor_id=1)

        self.assertTrue(demand.demand_active)
        self.assertEqual(demand.waiting_count, 1)
        self.assertEqual(demand.passing_count, 1)


class RequestManagerTests(unittest.TestCase):
    def test_duplicate_suppression_and_expiry(self) -> None:
        manager = RequestManager(cancel_after_sec=1.0)
        demand = FloorDemand(
            floor_id=1,
            timestamp=10.0,
            demand_active=True,
            dispatch_ready=True,
            priority="NORMAL",
            waiting_count=1,
            approaching_count=0,
            passing_count=0,
            crowd_score=1.0,
        )

        update = manager.update_from_demand(demand)
        self.assertEqual(len(update.created), 1)
        self.assertEqual(len(manager.get_assignable_calls()), 1)

        manager.update_from_demand(demand, now=10.5)
        expired = manager.expire_stale(11.1)
        self.assertEqual(expired, [])

        expired = manager.expire_stale(11.6)
        self.assertEqual(len(expired), 1)
        self.assertFalse(manager.has_active_calls())

    def test_service_cooldown_blocks_immediate_recreation(self) -> None:
        manager = RequestManager(service_cooldown_sec=2.0)
        demand = FloorDemand(
            floor_id=1,
            timestamp=20.0,
            demand_active=True,
            dispatch_ready=True,
            priority="NORMAL",
            waiting_count=1,
            approaching_count=0,
            passing_count=0,
            crowd_score=1.0,
        )

        manager.update_from_demand(demand)
        manager.mark_served(1, served_at=20.2)
        suppressed = manager.update_from_demand(demand, now=21.0)
        self.assertEqual(len(suppressed.created), 0)

        resumed = manager.update_from_demand(demand, now=22.5)
        self.assertEqual(len(resumed.created), 1)


class SchedulerTests(unittest.TestCase):
    def test_prefers_closer_idle_lift(self) -> None:
        scheduler = SchedulerModule(num_elevators=2, total_floors=10)
        hall_call = HallCall(
            floor_id=5,
            priority="NORMAL",
            created_at=0.0,
            last_seen_at=0.0,
        )
        states = {
            "Lift-1": LiftState(lift_id="Lift-1", current_floor=4, direction="IDLE"),
            "Lift-2": LiftState(
                lift_id="Lift-2",
                current_floor=8,
                direction="UP",
                scheduled_stops=[9],
            ),
        }

        assigned = scheduler.assign_elevator(states, hall_call)
        self.assertEqual(assigned, "Lift-1")


if __name__ == "__main__":
    unittest.main()
