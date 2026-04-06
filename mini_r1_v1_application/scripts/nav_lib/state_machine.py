"""Config-driven state machine for maze navigation."""
import json
import math
import time
from geometry_msgs.msg import Twist
from .sensor_state import SensorState


class ConfigStateMachine:
    def __init__(self, cfg, detectors, behaviors, logger):
        self.cfg = cfg
        self.detectors = detectors
        self.behaviors = behaviors
        self.logger = logger

        self.states_cfg = cfg.get('states', {})
        self.sign_dispatch = cfg.get('sign_dispatch', {})
        self.recovery_strategies = cfg.get('recovery_strategies', {})

        self.current_state = 'IDLE'
        self.current_behavior = None
        self.state_enter_time = 0.0

        # Recovery tracking
        self.active_recovery = None       # name of current recovery strategy
        self.recovery_step_idx = 0
        self.recovery_step_behavior = None

        # Sign approach tracking
        self.pending_sign_dir = None
        self.pending_sign_behavior = None
        self.sign_detect_x = 0.0
        self.sign_detect_y = 0.0

        # Stats
        self.signs_seen = 0
        self.arucos_seen = set()
        self.reasoning = "Initializing..."

    def tick(self, ss: SensorState) -> Twist:
        now = ss.now()

        # Update all detectors
        for det in self.detectors.values():
            det.update(ss)

        # Check transitions for current state
        state_cfg = self.states_cfg.get(self.current_state, {})
        transitions = state_cfg.get('transitions', [])

        for trans in transitions:
            det_name = trans.get('detector', '')

            # Pseudo-detectors
            if det_name == 'always':
                self._transition_to(trans, ss)
                break
            elif det_name == 'behavior_complete':
                if self.current_behavior and (
                    self.current_behavior.is_complete() or
                    self.current_behavior.is_timed_out(ss)):
                    self._transition_to(trans, ss)
                    break
            elif det_name == 'timer':
                dur = trans.get('params', {}).get('duration_s', 2.0)
                if (now - self.state_enter_time) >= dur:
                    self._transition_to(trans, ss)
                    break
            elif det_name == 'recovery_complete':
                if self._is_recovery_complete(ss):
                    self.reasoning = "Recovery complete, resuming exploration."
                    self._transition_to(trans, ss)
                    break
            elif det_name == 'recovery_failed':
                if self._is_recovery_failed(ss):
                    self.reasoning = "Recovery exhausted, trying exploration."
                    self._transition_to(trans, ss)
                    break
            elif det_name in self.detectors:
                det = self.detectors[det_name]
                if det.is_triggered():
                    self._handle_detector_transition(det_name, det, trans, ss)
                    break

        # SIGN_APPROACH: keep driving until we've moved close to sign area
        if self.current_state == 'SIGN_APPROACH':
            approach_time = now - self.state_enter_time
            # After 3 seconds of approach (or 1.5m traveled), execute the turn
            dx = ss.x - getattr(self, 'sign_detect_x', ss.x)
            dy = ss.y - getattr(self, 'sign_detect_y', ss.y)
            dist_traveled = math.sqrt(dx*dx + dy*dy)
            if approach_time > 4.5 or dist_traveled > 2.25:
                beh_name = getattr(self, 'pending_sign_behavior', None)
                sign_dir = getattr(self, 'pending_sign_dir', '?')
                if beh_name and beh_name in self.behaviors:
                    self.current_state = 'EXECUTING_SIGN'
                    self.state_enter_time = now
                    self.current_behavior = self.behaviors[beh_name]
                    self.current_behavior.reset()
                    self.current_behavior.start(ss)
                    self.reasoning = f"Reached sign area, turning {sign_dir}."
                    self.logger.info(f"State: SIGN_APPROACH → EXECUTING_SIGN ({sign_dir})")
                else:
                    self.current_state = 'EXPLORING'
                    self.state_enter_time = now

        # Execute current behavior
        if self.current_state == 'RECOVERING' and self.recovery_step_behavior:
            twist = self.recovery_step_behavior.tick(ss)
            if self.recovery_step_behavior.is_complete() or \
               self.recovery_step_behavior.is_timed_out(ss):
                self._advance_recovery(ss)
            return twist
        elif self.current_behavior:
            return self.current_behavior.tick(ss)

        return Twist()  # zero vel fallback

    def _transition_to(self, trans, ss: SensorState):
        next_state = trans.get('next', self.current_state)
        old_state = self.current_state
        self.current_state = next_state
        self.state_enter_time = ss.now()

        state_cfg = self.states_cfg.get(next_state, {})
        on_enter = state_cfg.get('on_enter')

        if on_enter and on_enter in self.behaviors:
            self.current_behavior = self.behaviors[on_enter]
            self.current_behavior.reset()
            self.current_behavior.start(ss)

        # Start recovery if specified
        recovery_name = trans.get('recovery')
        if recovery_name and next_state == 'RECOVERING':
            self._start_recovery(recovery_name, ss)

        if old_state != next_state:
            self.logger.info(f"State: {old_state} → {next_state}")

    def _handle_detector_transition(self, det_name, det, trans, ss):
        if det_name == 'sign_visible':
            sign_dir = det.get_value()
            self.signs_seen += 1
            self.pending_sign_dir = sign_dir
            self.pending_sign_behavior = self.sign_dispatch.get(sign_dir)
            # Record where the sign was detected from
            self.sign_detect_x = ss.x
            self.sign_detect_y = ss.y

            # Go to SIGN_APPROACH — keep driving forward until close
            self.current_state = 'SIGN_APPROACH'
            self.state_enter_time = ss.now()
            # Use gap_follow to keep navigating toward the sign
            if 'gap_follow' in self.behaviors:
                self.current_behavior = self.behaviors['gap_follow']
                self.current_behavior.reset()
                self.current_behavior.start(ss)
            self.reasoning = f"Sign '{sign_dir}' spotted — approaching before turning."
            self.logger.info(f"State: EXPLORING → SIGN_APPROACH (sign={sign_dir})")
            ss.last_sign = ""
            return

        if det_name == 'goal_reached':
            self.reasoning = "Goal reached! Mission complete."
        elif det_name == 'dead_end':
            self.reasoning = "Dead end detected. Starting recovery."
        elif det_name == 'stuck':
            self.reasoning = "Robot stuck. Starting recovery."
        elif det_name == 'loop':
            self.reasoning = "Loop detected. Breaking pattern."

        self._transition_to(trans, ss)

    def _start_recovery(self, strategy_name, ss):
        strategy = self.recovery_strategies.get(strategy_name, {})
        self.active_recovery = strategy_name
        self.recovery_step_idx = 0
        steps = strategy.get('steps', [])
        self.reasoning = f"Recovery: {strategy_name} step 1/{len(steps)}"
        self._start_recovery_step(ss)

    def _start_recovery_step(self, ss):
        strategy = self.recovery_strategies.get(self.active_recovery, {})
        steps = strategy.get('steps', [])
        if self.recovery_step_idx < len(steps):
            beh_name = steps[self.recovery_step_idx]
            if beh_name in self.behaviors:
                self.recovery_step_behavior = self.behaviors[beh_name]
                self.recovery_step_behavior.reset()
                self.recovery_step_behavior.start(ss)
                self.reasoning = (
                    f"Recovery: {self.active_recovery} "
                    f"step {self.recovery_step_idx+1}/{len(steps)} "
                    f"({beh_name})")

    def _advance_recovery(self, ss):
        strategy = self.recovery_strategies.get(self.active_recovery, {})
        steps = strategy.get('steps', [])
        recheck = strategy.get('recheck')

        # Re-evaluate the problem detector
        if recheck and recheck in self.detectors:
            self.detectors[recheck].update(ss)
            if not self.detectors[recheck].is_triggered():
                # Problem resolved
                self.recovery_step_behavior = None
                return

        self.recovery_step_idx += 1
        if self.recovery_step_idx < len(steps):
            self._start_recovery_step(ss)
        else:
            self.recovery_step_behavior = None

    def _is_recovery_complete(self, ss) -> bool:
        if self.active_recovery is None:
            return True
        strategy = self.recovery_strategies.get(self.active_recovery, {})
        recheck = strategy.get('recheck')
        if recheck and recheck in self.detectors:
            self.detectors[recheck].update(ss)
            if not self.detectors[recheck].is_triggered():
                return True
        return self.recovery_step_behavior is None and \
               self.recovery_step_idx >= len(strategy.get('steps', []))

    def _is_recovery_failed(self, ss) -> bool:
        if self.active_recovery is None:
            return False
        strategy = self.recovery_strategies.get(self.active_recovery, {})
        steps = strategy.get('steps', [])
        return self.recovery_step_idx >= len(steps) and \
               self.recovery_step_behavior is None

    # ── VLM Override Methods ──────────────────────────────────────────────
    def force_behavior(self, name, ss):
        """VLM brain forces a specific behavior, interrupting current state."""
        if name not in self.behaviors:
            self.logger.warn(f"VLM requested unknown behavior: {name}")
            return
        self.current_state = 'EXECUTING_SIGN'  # reuse this state
        self.state_enter_time = ss.now()
        self.current_behavior = self.behaviors[name]
        self.current_behavior.reset()
        self.current_behavior.start(ss)
        self.reasoning = f"VLM override: executing '{name}'"
        self.logger.info(f"VLM force_behavior: {name}")

    def force_recovery(self, name, ss):
        """VLM brain forces a recovery strategy."""
        if name not in self.recovery_strategies:
            self.logger.warn(f"VLM requested unknown recovery: {name}")
            return
        self.current_state = 'RECOVERING'
        self.state_enter_time = ss.now()
        self._start_recovery(name, ss)
        self.reasoning = f"VLM override: recovery '{name}'"
        self.logger.info(f"VLM force_recovery: {name}")

    def get_status_json(self, ss: SensorState) -> str:
        det_status = {}
        for name, det in self.detectors.items():
            if name == 'sign_visible' and det.is_triggered():
                det_status[name] = {
                    'active': True, 'value': det.get_value(),
                    'age_s': round(ss.now() - ss.sign_stamp, 1)}
            else:
                det_status[name] = det.is_triggered()

        beh_name = ""
        beh_elapsed = 0.0
        if self.current_state == 'RECOVERING' and self.recovery_step_behavior:
            beh_name = self.recovery_step_behavior.name
            beh_elapsed = self.recovery_step_behavior.elapsed(ss)
        elif self.current_behavior:
            beh_name = self.current_behavior.name
            beh_elapsed = self.current_behavior.elapsed(ss)

        return json.dumps({
            'state': self.current_state,
            'behavior': beh_name,
            'behavior_elapsed_s': round(beh_elapsed, 1),
            'detectors': det_status,
            'recovery': self.active_recovery if self.current_state == 'RECOVERING' else None,
            'signs_seen': self.signs_seen,
            'arucos_seen': sorted(self.arucos_seen),
            'position': {
                'x': round(ss.x, 2),
                'y': round(ss.y, 2),
                'yaw_deg': round(ss.yaw * 180 / 3.14159, 0)},
            'reasoning': self.reasoning,
        })
