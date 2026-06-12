"""
vision.py
Real-time OpenCV camera input with YOLO-based person tracking plus
simulation fallback for the smart lift MVP.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

from config import APPROACH_ZONE, DEFAULT_MODEL_PATH, TRACK_STALE_SEC, WAITING_ZONE
from contracts import PersonTrack, VisionSnapshot

try:
    import cv2  # type: ignore[import]

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO  # type: ignore[import]

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore[assignment]
    ULTRALYTICS_AVAILABLE = False

try:
    import lap  # type: ignore[import]  # noqa: F401

    LAP_AVAILABLE = True
except ImportError:
    LAP_AVAILABLE = False


Frame = Any | None


class VisionModule:
    """
    Wraps live video capture with a stable simulation fallback.
    """

    def __init__(
        self,
        camera_id: int | str = 0,
        model_path: str = DEFAULT_MODEL_PATH,
    ) -> None:
        self.camera_id = camera_id
        self.model_path = model_path
        self.model: Any | None = None
        self.cap: Any | None = None
        self.live_mode = False
        self.use_yolo_tracker = False
        self.track_memory: dict[str, dict[str, float]] = {}
        self.last_time = time.time()
        self._simulation_step = 0
        self._next_fallback_track_id = 1

        if CV2_AVAILABLE and ULTRALYTICS_AVAILABLE:
            try:
                print("[VisionModule] Loading YOLO model.")
                self.model = YOLO(self.model_path)
                self._open_camera()
                self.live_mode = True
                self.use_yolo_tracker = LAP_AVAILABLE
                print(f"[VisionModule] Live detection active on camera {self.camera_id}.")
                if not LAP_AVAILABLE:
                    print(
                        "[VisionModule] 'lap' is not installed. "
                        "Using built-in centroid tracking fallback."
                    )
            except Exception as exc:
                print(f"[VisionModule] Falling back to simulation mode: {exc}")
                self.model = None
                self.cap = None
                self.live_mode = False
        else:
            print("[VisionModule] OpenCV/Ultralytics unavailable. Running in simulation mode.")

    def _open_camera(self) -> None:
        if not CV2_AVAILABLE or cv2 is None:
            raise RuntimeError("OpenCV is not available.")

        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera '{self.camera_id}'. Ensure the camera is connected."
            )

    def _zone_contains(self, zone: dict[str, float], x_norm: float, y_norm: float) -> bool:
        return (
            zone["x_min"] <= x_norm <= zone["x_max"]
            and zone["y_min"] <= y_norm <= zone["y_max"]
        )

    def _cleanup_stale_tracks(self, current_time: float) -> None:
        stale_ids = [
            track_id
            for track_id, memory in self.track_memory.items()
            if current_time - memory["last_seen_at"] > TRACK_STALE_SEC
        ]
        for track_id in stale_ids:
            del self.track_memory[track_id]

    def _assign_fallback_track_id(
        self,
        x_norm: float,
        y_norm: float,
        current_time: float,
        claimed_ids: set[str],
    ) -> str:
        best_track_id: str | None = None
        best_distance = float("inf")

        for track_id, memory in self.track_memory.items():
            if track_id in claimed_ids:
                continue
            if current_time - memory["last_seen_at"] > TRACK_STALE_SEC:
                continue

            dx = x_norm - memory["x_norm"]
            dy = y_norm - memory["y_norm"]
            distance = math.hypot(dx, dy)
            if distance < 0.12 and distance < best_distance:
                best_distance = distance
                best_track_id = track_id

        if best_track_id is not None:
            claimed_ids.add(best_track_id)
            return best_track_id

        track_id = f"fallback-{self._next_fallback_track_id}"
        self._next_fallback_track_id += 1
        claimed_ids.add(track_id)
        return track_id

    def _build_track(
        self,
        track_id: str,
        x_center: float,
        y_center: float,
        bbox_height: float,
        frame_width: float,
        frame_height: float,
        current_time: float,
        dt: float,
    ) -> PersonTrack:
        x_norm = min(max(x_center / max(frame_width, 1.0), 0.0), 1.0)
        y_norm = min(max(y_center / max(frame_height, 1.0), 0.0), 1.0)
        waiting_center_x = (WAITING_ZONE["x_min"] + WAITING_ZONE["x_max"]) / 2.0
        waiting_center_y = (WAITING_ZONE["y_min"] + WAITING_ZONE["y_max"]) / 2.0

        distance_m = float(max(0.5, 1000.0 / (bbox_height + 1.0)))
        in_waiting_zone = self._zone_contains(WAITING_ZONE, x_norm, y_norm)
        in_approach_zone = (
            self._zone_contains(APPROACH_ZONE, x_norm, y_norm) and not in_waiting_zone
        )

        memory = self.track_memory.get(track_id)
        if memory is None:
            speed = 0.0
            motion_angle = math.pi / 2.0
            first_seen_at = current_time
        else:
            prev_x = memory["x_norm"]
            prev_y = memory["y_norm"]
            dx = x_norm - prev_x
            dy = y_norm - prev_y
            speed = math.hypot(dx, dy) / max(dt, 1e-3)

            to_door_x = waiting_center_x - x_norm
            to_door_y = waiting_center_y - y_norm
            movement_norm = math.hypot(dx, dy)
            target_norm = math.hypot(to_door_x, to_door_y)
            if movement_norm > 0 and target_norm > 0:
                cos_theta = ((dx * to_door_x) + (dy * to_door_y)) / (
                    movement_norm * target_norm
                )
                cos_theta = min(1.0, max(-1.0, cos_theta))
                motion_angle = math.acos(cos_theta)
            else:
                motion_angle = math.pi / 2.0
            first_seen_at = memory["first_seen_at"]

        self.track_memory[track_id] = {
            "x_norm": x_norm,
            "y_norm": y_norm,
            "first_seen_at": first_seen_at,
            "last_seen_at": current_time,
        }

        return PersonTrack(
            track_id=track_id,
            x_norm=x_norm,
            y_norm=y_norm,
            bbox_height=bbox_height,
            distance_m=distance_m,
            speed=speed,
            motion_angle=motion_angle,
            in_waiting_zone=in_waiting_zone,
            in_approach_zone=in_approach_zone,
            seen_for_sec=max(0.0, current_time - first_seen_at),
            last_seen_ts=current_time,
        )

    def _detect_without_tracker(
        self,
        frame: Any,
        current_time: float,
        dt: float,
    ) -> tuple[list[PersonTrack], Frame]:
        if self.model is None:
            return [], frame

        results = self.model.predict(
            frame,
            classes=[0],
            verbose=False,
        )

        tracks: list[PersonTrack] = []
        frame_height, frame_width = frame.shape[:2]
        claimed_ids: set[str] = set()

        if results and results[0].boxes:
            boxes = results[0].boxes
            xywh = boxes.xywh.cpu().numpy()  # type: ignore[union-attr]

            for box in xywh:
                x_center, y_center, _box_width, box_height = box
                x_norm = min(max(float(x_center) / max(float(frame_width), 1.0), 0.0), 1.0)
                y_norm = min(max(float(y_center) / max(float(frame_height), 1.0), 0.0), 1.0)
                track_id = self._assign_fallback_track_id(
                    x_norm=x_norm,
                    y_norm=y_norm,
                    current_time=current_time,
                    claimed_ids=claimed_ids,
                )
                track = self._build_track(
                    track_id=track_id,
                    x_center=float(x_center),
                    y_center=float(y_center),
                    bbox_height=float(box_height),
                    frame_width=float(frame_width),
                    frame_height=float(frame_height),
                    current_time=current_time,
                    dt=dt,
                )
                tracks.append(track)

        annotated = results[0].plot() if results else frame
        return tracks, annotated

    def _infer(self) -> tuple[VisionSnapshot, Frame]:
        if self.cap is None or self.model is None:
            return self._simulate()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[VisionModule] Frame grab failed. Switching to simulation output for this cycle.")
            return self._simulate()

        current_time = time.time()
        dt = max(current_time - self.last_time, 1e-3)
        self.last_time = current_time
        self._cleanup_stale_tracks(current_time)

        tracks: list[PersonTrack] = []
        annotated: Frame = frame

        if self.use_yolo_tracker:
            results = self.model.track(
                frame,
                classes=[0],
                persist=True,
                verbose=False,
            )

            frame_height, frame_width = frame.shape[:2]

            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes
                ids = boxes.id.cpu().numpy()  # type: ignore[union-attr]
                xywh = boxes.xywh.cpu().numpy()  # type: ignore[union-attr]

                for index, raw_track_id in enumerate(ids):  # type: ignore[misc]
                    x_center, y_center, _box_width, box_height = xywh[index]
                    track = self._build_track(
                        track_id=str(int(raw_track_id)),
                        x_center=float(x_center),
                        y_center=float(y_center),
                        bbox_height=float(box_height),
                        frame_width=float(frame_width),
                        frame_height=float(frame_height),
                        current_time=current_time,
                        dt=dt,
                    )
                    tracks.append(track)

            annotated = results[0].plot() if results else frame
        else:
            tracks, annotated = self._detect_without_tracker(frame, current_time, dt)

        snapshot = VisionSnapshot(timestamp=current_time, people=tracks, source="live")
        return snapshot, annotated

    def _simulation_people(self) -> list[tuple[str, float, float, float]]:
        patterns: list[list[tuple[str, float, float, float]]] = [
            [],
            [],
            [("1", 0.52, 0.28, 200.0)],
            [("1", 0.52, 0.40, 230.0)],
            [("1", 0.51, 0.54, 260.0)],
            [("1", 0.50, 0.66, 320.0)],
            [("1", 0.50, 0.67, 320.0)],
            [("1", 0.50, 0.68, 320.0), ("2", 0.63, 0.62, 300.0)],
            [
                ("1", 0.50, 0.68, 320.0),
                ("2", 0.63, 0.62, 300.0),
                ("3", 0.40, 0.64, 295.0),
            ],
            [("4", 0.10, 0.35, 180.0)],
            [],
            [],
        ]
        return patterns[self._simulation_step % len(patterns)]

    def _simulate(self) -> tuple[VisionSnapshot, Frame]:
        current_time = time.time()
        dt = max(current_time - self.last_time, 1e-3)
        self.last_time = current_time
        self._cleanup_stale_tracks(current_time)

        tracks: list[PersonTrack] = []
        for track_id, x_norm, y_norm, bbox_height in self._simulation_people():
            track = self._build_track(
                track_id=track_id,
                x_center=x_norm * 1000.0,
                y_center=y_norm * 1000.0,
                bbox_height=bbox_height,
                frame_width=1000.0,
                frame_height=1000.0,
                current_time=current_time,
                dt=dt,
            )
            tracks.append(track)

        if not tracks and random.random() < 0.10:
            drift_x = random.uniform(0.05, 0.12)
            drift_y = random.uniform(0.25, 0.45)
            track = self._build_track(
                track_id="bg",
                x_center=drift_x * 1000.0,
                y_center=drift_y * 1000.0,
                bbox_height=170.0,
                frame_width=1000.0,
                frame_height=1000.0,
                current_time=current_time,
                dt=dt,
            )
            tracks.append(track)

        self._simulation_step += 1
        snapshot = VisionSnapshot(timestamp=current_time, people=tracks, source="simulation")
        return snapshot, None

    def process_frame_with_frame(self) -> tuple[VisionSnapshot, Frame]:
        if self.live_mode:
            return self._infer()
        return self._simulate()

    def process_frame(self) -> VisionSnapshot:
        snapshot, _ = self.process_frame_with_frame()
        return snapshot

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[VisionModule] Camera released.")
