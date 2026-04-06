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
        if self._start_time < 1.0:
            return False  # not properly started yet
        return (ss.now() - self._start_time) > self.timeout_s

    def elapsed(self, ss: SensorState) -> float:
        if self._start_time < 1.0:
            return 0.0
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


class LaserGapFollowBehavior(BaseBehavior):
    """Follow the largest gap using raw LiDAR ranges — no costmap needed.

    LiDAR gives 360 range readings at 25Hz. We find contiguous arcs where
    range > safety_distance, score them with a forward bias, and steer
    toward the best gap's center. Much more responsive than costmap
    ray-marching (2Hz, grid artifacts).
    """
    def tick(self, ss: SensorState) -> Twist:
        if self.is_timed_out(ss):
            self._complete = True
            return self._clamp_twist(0.0, 0.0)

        if ss.lidar_ranges is None or len(ss.lidar_ranges) == 0:
            return self._clamp_twist(0.3, 0.0)

        ranges = np.array(ss.lidar_ranges, dtype=np.float32)
        n = len(ranges)

        safety = self.params.get('safety_distance_m', 0.4)
        min_gap_deg = self.params.get('min_gap_width_deg', 15)
        forward_bias = self.params.get('forward_bias', 1.5)
        kp = self.params.get('kp_angular', 1.2)
        slowdown_dist = self.params.get('slowdown_distance_m', 0.8)
        full_speed_dist = self.params.get('full_speed_distance_m', 2.0)

        # LiDAR is 360°: index 0 = angle_min, index n-1 = angle_max
        # For our LiDAR: angle_min=-3.14, angle_max=3.14, so index 0=behind,
        # index n/2=front. But let's compute properly:
        angle_min = ss.lidar_angle_min
        angle_inc = ss.lidar_angle_increment
        if angle_inc == 0:
            return self._clamp_twist(0.3, 0.0)

        # Forward index (angle closest to 0 = robot's front)
        forward_idx = int(-angle_min / angle_inc) if angle_inc > 0 else n // 2
        forward_idx = max(0, min(n - 1, forward_idx))

        # Mark each ray as "free" if range > safety and valid
        free = np.zeros(n, dtype=bool)
        for i in range(n):
            r = ranges[i]
            if np.isfinite(r) and r > safety:
                free[i] = True

        # Find contiguous gaps
        min_gap_samples = max(1, int(min_gap_deg / (360.0 / n)))
        gaps = []
        in_gap = False
        gap_start = 0
        for i in range(n):
            if free[i] and not in_gap:
                gap_start = i
                in_gap = True
            elif not free[i] and in_gap:
                if i - gap_start >= min_gap_samples:
                    gaps.append((gap_start, i - 1))
                in_gap = False
        if in_gap and n - gap_start >= min_gap_samples:
            gaps.append((gap_start, n - 1))

        if not gaps:
            # No gaps — rotate in place
            return self._clamp_twist(0.0, 0.5)

        # Score each gap: width * forward_bias_if_contains_forward
        best_gap = None
        best_score = -1
        for gs, ge in gaps:
            width = ge - gs + 1
            center = (gs + ge) // 2

            # Forward bias: how close is the gap center to forward?
            dist_to_forward = abs(center - forward_idx)
            # Normalize: 0 = exactly forward, 1 = opposite direction
            norm_dist = dist_to_forward / (n / 2)
            # Score: wider gaps are better, forward gaps get a bonus
            bias = forward_bias if norm_dist < 0.3 else 1.0
            score = width * bias

            # Also consider average range in the gap (deeper = better)
            gap_ranges = ranges[gs:ge+1]
            valid_ranges = gap_ranges[np.isfinite(gap_ranges)]
            if len(valid_ranges) > 0:
                avg_depth = np.mean(valid_ranges)
                score *= min(2.0, avg_depth / 1.0)  # bonus for deeper gaps

            if score > best_score:
                best_score = score
                best_gap = (gs, ge)

        # Pure pursuit: pick the target point at the deepest range
        # within the best gap, then steer toward it
        gs, ge = best_gap
        gap_ranges = ranges[gs:ge+1].copy()
        gap_ranges[~np.isfinite(gap_ranges)] = 0.0
        # Find the index within the gap with the deepest range
        best_local_idx = int(np.argmax(gap_ranges))
        target_idx = gs + best_local_idx
        target_angle = angle_min + target_idx * angle_inc
        target_range = float(gap_ranges[best_local_idx])

        # Clamp target range to a lookahead distance
        lookahead = self.params.get('lookahead_m', 1.5)
        target_range = min(target_range, lookahead)

        # Target point in robot frame (x=forward, y=left)
        tx = target_range * math.cos(target_angle)
        ty = target_range * math.sin(target_angle)

        # Pure pursuit: curvature = 2 * ty / L^2, where L = distance to target
        L_sq = tx * tx + ty * ty
        if L_sq < 0.01:
            return self._clamp_twist(0.1, 0.0)

        curvature = 2.0 * ty / L_sq
        angular = kp * curvature

        # Linear speed: scale by distance to nearest obstacle in forward cone
        cone_half = max(1, n // 36)
        fwd_lo = max(0, forward_idx - cone_half)
        fwd_hi = min(n, forward_idx + cone_half + 1)
        fwd_ranges = ranges[fwd_lo:fwd_hi]
        fwd_valid = fwd_ranges[np.isfinite(fwd_ranges) & (fwd_ranges > 0)]
        forward_clear = float(np.min(fwd_valid)) if len(fwd_valid) > 0 else 0.0

        if forward_clear < safety:
            linear = 0.0
        elif forward_clear < slowdown_dist:
            linear = self.max_linear * 0.25
        elif forward_clear < full_speed_dist:
            scale = (forward_clear - slowdown_dist) / (full_speed_dist - slowdown_dist)
            linear = self.max_linear * (0.25 + 0.75 * scale)
        else:
            linear = self.max_linear

        return self._clamp_twist(linear, angular)


BEHAVIOR_REGISTRY = {
    'timed_twist': TimedTwistBehavior,
    'proportional_turn': ProportionalTurnBehavior,
    'random_turn_forward': RandomTurnForwardBehavior,
    'laser_gap_follow': LaserGapFollowBehavior,
}
