"""Centralized sensor state consumed by all detectors and behaviors."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SensorState:
    # Robot pose from TF (odom → base_link)
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    odom_stamp: float = 0.0

    # Position history for loop detection: (x, y, timestamp)
    position_history: list = field(default_factory=list)
    last_position_log_time: float = 0.0

    # Costmap (raw OccupancyGrid data)
    costmap_data: np.ndarray = None       # flat array of costs
    costmap_width: int = 0
    costmap_height: int = 0
    costmap_resolution: float = 0.05
    costmap_origin_x: float = 0.0
    costmap_origin_y: float = 0.0
    costmap_stamp: float = 0.0

    # Sign detection
    last_sign: str = ""
    sign_stamp: float = 0.0

    # Mission status
    mission_complete: bool = False

    # LiDAR ranges (for wall follow)
    lidar_ranges: np.ndarray = None
    lidar_angle_min: float = 0.0
    lidar_angle_increment: float = 0.0
    lidar_stamp: float = 0.0

    def now(self) -> float:
        """Return the most recent stamp as a rough 'now'."""
        return max(self.odom_stamp, self.costmap_stamp, self.lidar_stamp, 0.001)
