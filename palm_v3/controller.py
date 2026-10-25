"""Timed gesture controller. Pure logic: this module never sends a key."""
from dataclasses import dataclass
import math


SHORTCUTS = {'next': 'right', 'previous': 'left', 'blackout': 'b'}


@dataclass(frozen=True)
class ControllerSettings:
    score_threshold: float = .85
    gesture_hold_seconds: float = .5
    neutral_hold_seconds: float = .3
    minimum_action_interval_seconds: float = .8
    arming_countdown_seconds: float = 5.
    maximum_prediction_gap_seconds: float = .25

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid controller setting: {name}')
        if not 0 <= self.score_threshold <= 1:
            raise ValueError('Score threshold must be between zero and one')
        if min(self.gesture_hold_seconds, self.neutral_hold_seconds, self.maximum_prediction_gap_seconds) <= 0:
            raise ValueError('Hold times and maximum prediction gap must be positive')
        if min(self.minimum_action_interval_seconds, self.arming_countdown_seconds) < 0:
            raise ValueError('Countdown and action interval cannot be negative')

    @classmethod
    def from_config(cls, config):
        supplied = config.get('controller', {})
        if not isinstance(supplied, dict):
            raise ValueError('Missing controller configuration')
        names = ('score_threshold', 'gesture_hold_seconds', 'neutral_hold_seconds',
                 'minimum_action_interval_seconds', 'arming_countdown_seconds')
        return cls(**{name: supplied[name] for name in names})


@dataclass(frozen=True)
class Action:
    gesture: str
    key: str
    at: float
    held_seconds: float


class GestureController:
    def __init__(self, settings):
        self.settings = settings
        self.phase = 'DISARMED'
        self.deadline = None
        self.candidate = None
        self.candidate_since = None
        self.last_action_at = None
        self.last_update_at = None

    def _clear_hold(self):
        self.candidate = self.candidate_since = None

    def request_arm(self, now):
        self._validate_time(now)
        self.phase = 'ARMING'
        self.deadline = now + self.settings.arming_countdown_seconds
        self._clear_hold()

    def disarm(self):
        self.phase = 'DISARMED'
        self.deadline = None
        self._clear_hold()

    def _validate_time(self, now):
        if not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError('Controller time must be finite')
        if self.last_update_at is not None and now < self.last_update_at:
            raise ValueError('Controller time cannot go backward')

    def update(self, now, prediction):
        self._validate_time(now)
        gap = self.last_update_at is not None and now - self.last_update_at > self.settings.maximum_prediction_gap_seconds
        self.last_update_at = now
        if gap:
            self._clear_hold()
            if self.phase == 'READY':
                self.phase = 'NEED NEUTRAL'
        if self.phase == 'DISARMED':
            return None
        if self.phase == 'ARMING':
            if now + 1e-9 < self.deadline:
                return None
            self.phase = 'NEED NEUTRAL'
            self._clear_hold()
        if prediction is None:
            self.phase = 'NEED NEUTRAL'
            self._clear_hold()
            return None
        label, score = prediction.label, prediction.score
        qualified = (label in ('neutral', *SHORTCUTS) and isinstance(score, (int, float))
                     and math.isfinite(score) and self.settings.score_threshold <= score <= 1)
        if not qualified:
            self._clear_hold()
            return None
        if label != self.candidate:
            self.candidate, self.candidate_since = label, now
        held = now - self.candidate_since
        if self.phase == 'NEED NEUTRAL':
            if label == 'neutral' and held + 1e-9 >= self.settings.neutral_hold_seconds:
                self.phase = 'READY'
                self._clear_hold()
            return None
        if label == 'neutral':
            return None
        interval_ok = (self.last_action_at is None or
                       now - self.last_action_at + 1e-9 >= self.settings.minimum_action_interval_seconds)
        if held + 1e-9 >= self.settings.gesture_hold_seconds and interval_ok:
            action = Action(label, SHORTCUTS[label], now, held)
            self.last_action_at = now
            self.phase = 'NEED NEUTRAL'
            self._clear_hold()
            return action
        return None

    def hint(self, now):
        if self.phase == 'DISARMED':
            return 'Press A here, then click the PowerPoint slideshow'
        if self.phase == 'ARMING':
            return f'Click PowerPoint now: {max(0, self.deadline-now):.1f}s'
        if self.phase == 'NEED NEUTRAL':
            return 'Show open palm to prepare one action'
        if self.candidate in SHORTCUTS:
            return f'Hold {self.candidate}: {max(0, now-self.candidate_since):.1f}/{self.settings.gesture_hold_seconds:g}s'
        return 'Ready: index up / V / fist'
