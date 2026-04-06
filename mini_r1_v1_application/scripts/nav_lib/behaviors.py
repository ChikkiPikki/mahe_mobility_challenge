"""Behavior registry — atomic action executors."""
import math
import random
import numpy as np
from geometry_msgs.msg import Twist
from .sensor_state import SensorState


def normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class BaseBehavior:
    def __init__(self, name, params, speed_profile, completion):
        self.name = name
        self.params = params
        self.max_linear = speed_profile.get('linear', 0.8)
        self.max_angular = speed_profile.get('angular', 0.5)
        self.completion = completion
        self.timeout_s = params.get('_timeout_s', 30.0)
        self._started = False
        self._start_time = 0.0
        self._complete = False

    def start(self, ss: SensorState):
        self._started = True
        self._start_time = ss.now()
        self._complete = False

    def tick(self, ss: SensorState) -> Twist:
        raise NotImplementedError

    def is_complete(self) -> bool:
        return self._complete

    def is_timed_out(self, ss: SensorState) -> bool:
        return (ss.now() - self._start_time) > self.timeout_s

    def elapsed(self, ss: SensorState) -> float:
        return ss.now() - self._start_time

    def reset(self):
        self._started = False
        self._complete = False

    def _clamp_twist(self, linear, angular) -> Twist:
        t = Twist()
        t.linear.x = max(-abs(self.max_linear), min(abs(self.max_linear), linear))
        t.angular.z = max(-abs(self.max_angular), min(abs(self.max_angular), angular))
        return t


class TimedTwistBehavior(BaseBehavior):
    """Publish a fixed twist for a duration or continuously."""
    def tick(self, ss: SensorState) -> Twist:
        method = self.completion.get('method', 'continuous')
        if method == 'duration':
            dur = self.completion.get('duration_s', 1.0)
            if self.elapsed(ss) >= dur:
                self._complete = True
        if self.is_timed_out(ss):
            self._complete = True
        return self._clamp_twist(
            self.params.get('linear_x', 0.0),
            self.params.get('angular_z', 0.0))


class ProportionalTurnBehavior(BaseBehavior):
    """Proportional controller to rotate by a target angle."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_yaw = 0.0
        self._initial_yaw = 0.0

    def start(self, ss: SensorState):
        super().start(ss)
        self._initial_yaw = ss.yaw
        delta = math.radians(self.params.get('target_angle_deg', 90))
        self._target_yaw = normalize_angle(self._initial_yaw + delta)

    def tick(self, ss: SensorState) -> Twist:
        error = normalize_angle(self._target_yaw - ss.yaw)
        tol = math.radians(self.params.get('tolerance_deg', 5))
        kp = self.params.get('kp', 2.0)

        if abs(error) < tol or self.is_timed_out(ss):
            self._complete = True
            return self._clamp_twist(0.0, 0.0)

        angular = kp * error
        return self._clamp_twist(0.0, angular)


class RandomTurnForwardBehavior(BaseBehavior):
    """Random turn then drive forward."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phase = 'turn'  # 'turn' or 'forward'
        self._target_yaw = 0.0
        self._forward_start = 0.0

    def start(self, ss: SensorState):
        super().start(ss)
        rng = self.params.get('turn_range_deg', [60, 180])
        angle = random.uniform(rng[0], rng[1])
        if random.random() < 0.5:
            angle = -angle
        self._target_yaw = normalize_angle(ss.yaw + math.radians(angle))
        self._phase = 'turn'

    def tick(self, ss: SensorState) -> Twist:
        if self.is_timed_out(ss):
            self._complete = True
            return self._clamp_twist(0.0, 0.0)

        if self._phase == 'turn':
            error = normalize_angle(self._target_yaw - ss.yaw)
            if abs(error) < math.radians(8):
                self._phase = 'forward'
                self._forward_start = ss.now()
                return self._clamp_twist(0.0, 0.0)
            return self._clamp_twist(0.0, 2.0 * error)

        # forward phase
        fwd_dur = self.params.get('forward_duration_s', 2.0)
        if ss.now() - self._forward_start >= fwd_dur:
            self._complete = True
            return self._clamp_twist(0.0, 0.0)
        return self._clamp_twist(0.4, 0.0)


class GapFollowBehavior(BaseBehavior):
    """Follow the largest gap in the costmap — primary corridor navigation."""
    def tick(self, ss: SensorState) -> Twist:
        if self.is_timed_out(ss):
            self._complete = True
            return self._clamp_twist(0.0, 0.0)

        if ss.costmap_data is None:
            return self._clamp_twist(0.2, 0.0)

        arc_half = math.radians(self.params.get('scan_arc_deg', 180) / 2.0)
        scan_min = self.params.get('scan_min_m', 0.3)
        scan_max = self.params.get('scan_max_m', 2.0)
        kp = self.params.get('kp_angular', 1.5)
        res = ss.costmap_resolution

        n_rays = 36
        angles = np.linspace(ss.yaw - arc_half, ss.yaw + arc_half, n_rays)
        free_dist = np.zeros(n_rays)

        for i, angle in enumerate(angles):
            for d in np.arange(scan_min, scan_max, res * 2):
                wx = ss.x + d * math.cos(angle)
                wy = ss.y + d * math.sin(angle)
                gx = int((wx - ss.costmap_origin_x) / res)
                gy = int((wy - ss.costmap_origin_y) / res)
                if 0 <= gx < ss.costmap_width and 0 <= gy < ss.costmap_height:
                    if ss.costmap_data[gy * ss.costmap_width + gx] >= 65:
                        break
                else:
                    break
                free_dist[i] = d

        # Find the widest contiguous gap
        best_start = 0
        best_len = 0
        cur_start = 0
        min_gap = self.params.get('min_gap_width_deg', 15)
        min_gap_rays = max(1, int(min_gap / (180.0 / n_rays)))

        for i in range(n_rays):
            if free_dist[i] > scan_min * 1.5:
                if i == 0 or free_dist[i-1] <= scan_min * 1.5:
                    cur_start = i
                gap_len = i - cur_start + 1
                if gap_len > best_len:
                    best_start = cur_start
                    best_len = gap_len

        if best_len < min_gap_rays:
            # No gap found — slow down, slight turn
            return self._clamp_twist(0.1, 0.3)

        # Steer toward the center of the best gap
        center_idx = best_start + best_len // 2
        target_angle = angles[center_idx]
        angle_error = normalize_angle(target_angle - ss.yaw)
        angular = kp * angle_error

        # Scale linear speed by forward clearance
        forward_idx = n_rays // 2
        forward_clear = free_dist[forward_idx] if forward_idx < n_rays else scan_min
        speed_scale = min(1.0, forward_clear / scan_max)
        linear = self.max_linear * max(0.15, speed_scale)

        return self._clamp_twist(linear, angular)


BEHAVIOR_REGISTRY = {
    'timed_twist': TimedTwistBehavior,
    'proportional_turn': ProportionalTurnBehavior,
    'random_turn_forward': RandomTurnForwardBehavior,
    'gap_follow': GapFollowBehavior,
}
