"""Detector registry — condition evaluators driven by sensor data."""
import math
import numpy as np
from .sensor_state import SensorState


class BaseDetector:
    def __init__(self, name, params, thresholds):
        self.name = name
        self.params = params
        self.thresholds = thresholds
        self._triggered = False
        self._value = None
        self._last_fire_time = 0.0
        self.cooldown_s = params.get('_cooldown_s', 0.0)

    def update(self, ss: SensorState):
        raise NotImplementedError

    def is_triggered(self) -> bool:
        return self._triggered

    def get_value(self):
        return self._value

    def reset(self):
        self._triggered = False
        self._value = None

    def _check_cooldown(self, now: float) -> bool:
        if now - self._last_fire_time < self.cooldown_s:
            return False
        return True

    def _fire(self, now: float, value=None):
        self._triggered = True
        self._value = value
        self._last_fire_time = now


class TopicMatchDetector(BaseDetector):
    """Fires when mission_complete is True."""
    def update(self, ss: SensorState):
        self._triggered = False
        if ss.mission_complete:
            self._fire(ss.now(), "MISSION_COMPLETE")


class TopicStringDetector(BaseDetector):
    """Fires when a valid sign string is received and not stale."""
    def update(self, ss: SensorState):
        self._triggered = False
        valid = self.params.get('valid_values', [])
        stale = self.params.get('stale_after_s', 2.0)
        now = ss.now()
        if not self._check_cooldown(now):
            return
        if ss.last_sign in valid and (now - ss.sign_stamp) < stale:
            self._fire(now, ss.last_sign)


class ArcOccupancyDetector(BaseDetector):
    """Fires when forward arc is mostly occupied (dead end)."""
    def update(self, ss: SensorState):
        self._triggered = False
        if ss.costmap_data is None:
            return
        now = ss.now()
        if not self._check_cooldown(now):
            return

        arc_half = math.radians(self.params.get('arc_half_deg', 60))
        check_dist = self.params.get('check_distance_m', 0.6)
        occ_ratio = self.params.get('occupied_ratio', 0.8)
        res = ss.costmap_resolution

        total = 0
        occupied = 0
        for angle_offset in np.linspace(-arc_half, arc_half, 30):
            angle = ss.yaw + angle_offset
            for d in np.arange(res, check_dist, res):
                wx = ss.x + d * math.cos(angle)
                wy = ss.y + d * math.sin(angle)
                gx = int((wx - ss.costmap_origin_x) / res)
                gy = int((wy - ss.costmap_origin_y) / res)
                if 0 <= gx < ss.costmap_width and 0 <= gy < ss.costmap_height:
                    cost = ss.costmap_data[gy * ss.costmap_width + gx]
                    total += 1
                    if cost >= 65:
                        occupied += 1

        if total > 0 and (occupied / total) > occ_ratio:
            self._fire(now)


class DisplacementWindowDetector(BaseDetector):
    """Fires when robot hasn't moved enough in the time window."""
    def __init__(self, name, params, thresholds):
        super().__init__(name, params, thresholds)
        self._history = []  # (x, y, time)

    def update(self, ss: SensorState):
        self._triggered = False
        now = ss.now()
        self._history.append((ss.x, ss.y, now))

        window = self.params.get('window_s', 4.0)
        disp_thresh = self.params.get('displacement_m', 0.05)

        # Prune old entries
        self._history = [(x, y, t) for x, y, t in self._history
                         if now - t <= window]

        if len(self._history) < 2:
            return
        if not self._check_cooldown(now):
            return

        oldest = self._history[0]
        dx = ss.x - oldest[0]
        dy = ss.y - oldest[1]
        displacement = math.sqrt(dx*dx + dy*dy)
        elapsed = now - oldest[2]

        if elapsed >= window * 0.8 and displacement < disp_thresh:
            self._fire(now)


class PositionRevisitDetector(BaseDetector):
    """Fires when robot is near a position visited long ago (loop)."""
    def update(self, ss: SensorState):
        self._triggered = False
        radius = self.params.get('radius_m', 0.5)
        min_gap = self.params.get('min_gap_s', 20.0)
        now = ss.now()
        if not self._check_cooldown(now):
            return

        for (hx, hy, ht) in ss.position_history:
            dx = ss.x - hx
            dy = ss.y - hy
            dist = math.sqrt(dx*dx + dy*dy)
            time_gap = now - ht
            if dist < radius and time_gap > min_gap:
                self._fire(now)
                return


class LateralGapDetector(BaseDetector):
    """Fires when there's an opening to the left or right in the costmap."""
    def update(self, ss: SensorState):
        self._triggered = False
        if ss.costmap_data is None:
            return
        now = ss.now()
        if not self._check_cooldown(now):
            return

        side = self.params.get('side', 'left')
        arc_half = math.radians(self.params.get('arc_half_deg', 45))
        min_dist = self.params.get('min_distance_m', 0.8)
        free_ratio = self.params.get('free_ratio', 0.5)
        res = ss.costmap_resolution

        # Center angle: left = yaw + 90°, right = yaw - 90°
        center = ss.yaw + (math.pi / 2 if side == 'left' else -math.pi / 2)

        total = 0
        free = 0
        for angle_offset in np.linspace(-arc_half, arc_half, 20):
            angle = center + angle_offset
            wx = ss.x + min_dist * math.cos(angle)
            wy = ss.y + min_dist * math.sin(angle)
            gx = int((wx - ss.costmap_origin_x) / res)
            gy = int((wy - ss.costmap_origin_y) / res)
            if 0 <= gx < ss.costmap_width and 0 <= gy < ss.costmap_height:
                cost = ss.costmap_data[gy * ss.costmap_width + gx]
                total += 1
                if cost < 65:
                    free += 1

        if total > 0 and (free / total) > free_ratio:
            self._fire(now)


# Registry mapping method names to classes
DETECTOR_REGISTRY = {
    'topic_match': TopicMatchDetector,
    'topic_string': TopicStringDetector,
    'arc_occupancy': ArcOccupancyDetector,
    'displacement_window': DisplacementWindowDetector,
    'position_revisit': PositionRevisitDetector,
    'lateral_gap': LateralGapDetector,
}
