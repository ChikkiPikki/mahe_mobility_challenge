"""Load and validate behavior_config.yaml."""
import yaml
from .detectors import DETECTOR_REGISTRY
from .behaviors import BEHAVIOR_REGISTRY


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    validate(cfg)
    return cfg


def validate(cfg: dict):
    assert cfg.get('config_version') == 1, "Unsupported config version"
    assert 'speed_profiles' in cfg
    assert 'thresholds' in cfg
    assert 'detectors' in cfg
    assert 'behaviors' in cfg
    assert 'states' in cfg

    for name, det in cfg['detectors'].items():
        method = det.get('method')
        assert method in DETECTOR_REGISTRY, f"Unknown detector method: {method}"

    for name, beh in cfg['behaviors'].items():
        btype = beh.get('type')
        assert btype in BEHAVIOR_REGISTRY, f"Unknown behavior type: {btype}"


def build_detectors(cfg: dict) -> dict:
    detectors = {}
    thresholds = cfg.get('thresholds', {})
    for name, det_cfg in cfg['detectors'].items():
        cls = DETECTOR_REGISTRY[det_cfg['method']]
        params = dict(det_cfg.get('params', {}))
        params['_cooldown_s'] = det_cfg.get('cooldown_s', 0.0)
        detectors[name] = cls(name, params, thresholds)
    return detectors


def build_behaviors(cfg: dict) -> dict:
    behaviors = {}
    profiles = cfg.get('speed_profiles', {})
    for name, beh_cfg in cfg['behaviors'].items():
        cls = BEHAVIOR_REGISTRY[beh_cfg['type']]
        profile = profiles.get(beh_cfg.get('speed_profile', 'corridor'), {})
        params = dict(beh_cfg.get('params', {}))
        params['_timeout_s'] = beh_cfg.get('timeout_s', 30.0)
        completion = beh_cfg.get('completion', {'method': 'continuous'})
        behaviors[name] = cls(name, params, profile, completion)
    return behaviors
