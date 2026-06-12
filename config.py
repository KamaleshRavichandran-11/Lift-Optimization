"""
config.py
System configuration constants and policy weights.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Intent thresholds
THRESHOLD = 0.45
APPROACH_THRESHOLD = 0.30
WAITING_DWELL_SEC = 1.5
TRACK_STALE_SEC = 1.0
CALL_CANCEL_EMPTY_SEC = 2.5
CALL_REFRESH_SEC = 0.5
SERVICE_COOLDOWN_SEC = 3.0

# Elevator configuration
MAX_CAPACITY = 10
TOTAL_FLOORS = 10
NUM_ELEVATORS = 2
DOOR_OPEN_TICKS = 2

# Elevator assignment cost weights
W1_DISTANCE = 1.0
W2_STOPS = 2.0
W3_LOAD = 1.5
W4_DIRECTION = 1.2
W5_IDLE = 0.5

# Intent predictor weights
W1_VELOCITY = 0.5
W2_DIRECTION = 0.3
W3_DISTANCE = 0.2
W4_WAITING_ZONE = 0.4
W5_DWELL = 0.3

# Demand scoring
ALPHA_PEOPLE = 1.0
BETA_TIME = 0.5
CROWD_THRESHOLD = 4
RUSH_HOUR_THRESHOLD = 6

# Idle policy
ENERGY_IDLE_SEC = 8.0
PARKING_FLOORS = [1, TOTAL_FLOORS]

# Normalized camera zones
WAITING_ZONE = {
    "x_min": 0.30,
    "x_max": 0.70,
    "y_min": 0.50,
    "y_max": 0.95,
}
APPROACH_ZONE = {
    "x_min": 0.15,
    "x_max": 0.85,
    "y_min": 0.20,
    "y_max": 0.70,
}

# Logging
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
ANALYTICS_LOG_PATH = os.path.join(BASE_DIR, "logs", "analytics.csv")
