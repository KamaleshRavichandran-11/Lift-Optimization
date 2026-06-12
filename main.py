"""
main.py
Smart vision-assisted elevator dispatch runtime.
"""

from __future__ import annotations

import time

from analytics import AnalyticsLogger
from config import ANALYTICS_LOG_PATH, APPROACH_ZONE, NUM_ELEVATORS, PARKING_FLOORS, TOTAL_FLOORS, WAITING_ZONE
from controller import ElevatorController
from contracts import FloorDemand, HallCall
from decision import DecisionModule
from request_manager import RequestManager
from scheduler import SchedulerModule
from vision import CV2_AVAILABLE, ULTRALYTICS_AVAILABLE, VisionModule, cv2


CAMERA_ID = 0
FLOOR_ID = 1
FRAME_DELAY_SEC = 0.1
SHOW_WINDOW = True


def build_elevators() -> dict[str, ElevatorController]:
    elevators: dict[str, ElevatorController] = {}

    for index in range(1, NUM_ELEVATORS + 1):
        parking_floor = PARKING_FLOORS[(index - 1) % len(PARKING_FLOORS)]
        elevators[f"Lift-{index}"] = ElevatorController(
            lift_id=f"Lift-{index}",
            initial_floor=parking_floor,
            parking_floor=parking_floor,
        )

    return elevators


def print_header() -> None:
    print("=" * 72)
    print("   Smart Vision-Assisted Elevator Dispatch System")
    print(f"   Camera: {CAMERA_ID} | Floor: {FLOOR_ID} | Elevators: {NUM_ELEVATORS}")
    print("=" * 72)
    if ULTRALYTICS_AVAILABLE:
        print("[INFO] YOLO tracking is available.")
    else:
        print("[WARN] YOLO tracking unavailable. Simulation mode will be used.")
    if not CV2_AVAILABLE:
        print("[WARN] OpenCV window support is unavailable.")
    print("       Press Ctrl-C or Q in the preview window to stop.\n")


def draw_zone_overlay(frame: object) -> None:
    if not CV2_AVAILABLE or cv2 is None or frame is None:
        return

    frame_height, frame_width = frame.shape[:2]
    for zone, color, label in (
        (APPROACH_ZONE, (255, 180, 0), "Approach"),
        (WAITING_ZONE, (0, 255, 80), "Waiting"),
    ):
        x1 = int(zone["x_min"] * frame_width)
        y1 = int(zone["y_min"] * frame_height)
        x2 = int(zone["x_max"] * frame_width)
        y2 = int(zone["y_max"] * frame_height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def print_cycle_summary(
    frame_number: int,
    source: str,
    demand: FloorDemand,
    active_calls: list[HallCall],
) -> None:
    print(
        f"\n[Frame {frame_number:04d}] source={source} "
        f"waiting={demand.waiting_count} approaching={demand.approaching_count} "
        f"passing={demand.passing_count} priority={demand.priority}"
    )
    print(
        f"  Demand -> active={demand.demand_active} dispatch_ready={demand.dispatch_ready} "
        f"crowd_score={demand.crowd_score:.2f}"
    )
    if active_calls:
        call_descriptions = [
            f"floor={call.floor_id} status={call.status} lift={call.assigned_lift_id or '-'}"
            for call in active_calls
        ]
        print(f"  Calls  -> {', '.join(call_descriptions)}")
    else:
        print("  Calls  -> none")


def run_live(max_frames: int | None = None) -> None:
    print_header()

    vision = VisionModule(camera_id=CAMERA_ID)
    decision = DecisionModule()
    scheduler = SchedulerModule(NUM_ELEVATORS, TOTAL_FLOORS)
    request_manager = RequestManager()
    analytics = AnalyticsLogger(ANALYTICS_LOG_PATH)
    elevators = build_elevators()

    frame_number = 0
    last_demand_ts = time.time()

    try:
        while True:
            frame_number += 1
            snapshot, annotated_frame = vision.process_frame_with_frame()
            demand = decision.analyze_snapshot(snapshot, FLOOR_ID)
            now = snapshot.timestamp

            request_update = request_manager.update_from_demand(demand, now)
            for call in request_update.created:
                analytics.log_event(
                    "call_created",
                    {
                        "floor_id": call.floor_id,
                        "priority": call.priority,
                        "people_count": call.people_count,
                    },
                )

            expired_calls = request_manager.expire_stale(now)
            for call in expired_calls:
                if call.assigned_lift_id and call.assigned_lift_id in elevators:
                    elevators[call.assigned_lift_id].cancel_stop(call.floor_id)
                analytics.log_event(
                    "call_expired",
                    {
                        "floor_id": call.floor_id,
                        "assigned_lift_id": call.assigned_lift_id,
                    },
                )

            if demand.demand_active:
                last_demand_ts = now

            for hall_call in request_manager.get_assignable_calls():
                assigned = scheduler.assign_elevator(
                    {lift_id: elevator.get_state() for lift_id, elevator in elevators.items()},
                    hall_call,
                )
                if assigned is None:
                    continue

                if not elevators[assigned].has_stop(hall_call.floor_id):
                    elevators[assigned].add_stop(hall_call.floor_id)
                request_manager.mark_assigned(hall_call.floor_id, assigned)
                analytics.log_event(
                    "call_assigned",
                    {
                        "floor_id": hall_call.floor_id,
                        "assigned_lift_id": assigned,
                    },
                )

            parking_actions = scheduler.recommend_parking_actions(
                {lift_id: elevator.get_state() for lift_id, elevator in elevators.items()},
                has_active_calls=request_manager.has_active_calls(),
                seconds_since_demand=max(0.0, now - last_demand_ts),
            )
            for lift_id, parking_floor in parking_actions.items():
                elevators[lift_id].park_at(parking_floor)

            service_events = []
            for elevator in elevators.values():
                service_event = elevator.move()
                if service_event is not None:
                    service_events.append(service_event)

            for event in service_events:
                served_call = request_manager.mark_served(event.floor_id, served_at=event.timestamp)
                analytics.log_event(
                    "call_served",
                    {
                        "floor_id": event.floor_id,
                        "lift_id": event.lift_id,
                        "matched_request": served_call is not None,
                    },
                )

            active_calls = request_manager.active_calls()
            print_cycle_summary(frame_number, snapshot.source, demand, active_calls)

            for lift_id, elevator in elevators.items():
                state = elevator.get_state()
                print(
                    f"  [{lift_id}] floor={state.current_floor:>2} dir={state.direction:<4} "
                    f"status={state.status:<10} stops={state.scheduled_stops}"
                )

            analytics.log_frame(
                frame_number,
                snapshot,
                demand,
                active_calls,
                {lift_id: elevator.get_state() for lift_id, elevator in elevators.items()},
            )

            if SHOW_WINDOW and CV2_AVAILABLE and annotated_frame is not None and cv2 is not None:
                draw_zone_overlay(annotated_frame)
                cv2.putText(
                    annotated_frame,
                    (
                        f"Floor {FLOOR_ID} | Waiting {demand.waiting_count} | "
                        f"Approaching {demand.approaching_count}"
                    ),
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 80),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Smart Lift Detection - Press Q to quit", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] 'q' pressed - shutting down.")
                    break
            else:
                time.sleep(FRAME_DELAY_SEC)

            if max_frames is not None and frame_number >= max_frames:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt - shutting down.")

    finally:
        analytics.close()
        vision.release()
        if SHOW_WINDOW and CV2_AVAILABLE and cv2 is not None:
            cv2.destroyAllWindows()
        print("[INFO] Resources released. Goodbye.")


if __name__ == "__main__":
    run_live()
