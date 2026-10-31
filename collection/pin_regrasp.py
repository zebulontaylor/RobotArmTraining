"""Execute release, retreat, return, and verified regrasp through a robot adapter."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


PHASE_NAMES = ("release", "retreat", "return", "close", "lift", "verify")


class PinBackend(Protocol):
    """Blocking control interface; poses use metres in one fixed workspace frame.

    ``command`` executes one interval at ``hz`` with local IK, joint, workspace,
    tracking, and collision checks. It raises RuntimeError when execution fails.
    Opening is per-finger displacement in metres, not full jaw separation.
    Observations are synchronized numeric arrays copied by the collector.
    ``reset`` prepares a table-supported grasp and returns serializable metadata;
    real hardware must implement physical placement/localization instead of
    simulation state assignment. ``verify_lift`` checks the object, not just FK.
    """

    hz: float
    open_width: float
    closed_width: float

    def reset(self, seed: int) -> dict: ...
    def observe(self) -> dict: ...
    def command(self, position: np.ndarray, rotation: np.ndarray, opening: float) -> None: ...
    def grasped(self) -> bool: ...
    def verify_lift(self, min_height: float) -> bool: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class CollectionConfig:
    retreat_min_m: float = 0.02
    retreat_max_m: float = 0.06
    speed_m_s: float = 0.025
    release_s: float = 0.5
    close_s: float = 0.7
    verify_s: float = 0.3
    lift_m: float = 0.03
    retreat_cone_deg: float = 8.0
    rotation_jitter_deg: float = 3.0
    angular_speed_deg_s: float = 15.0

    def __post_init__(self):
        values = np.array(list(vars(self).values()), dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Collection distances, speeds, and durations must be finite and positive")
        if self.retreat_min_m > self.retreat_max_m:
            raise ValueError("retreat_min_m must not exceed retreat_max_m")
        if max(self.retreat_cone_deg, self.rotation_jitter_deg) > 30:
            raise ValueError("Approach and wrist variations must not exceed 30 degrees")


def rotation_exp(vector):
    """Map an axis-angle vector in radians to a proper rotation matrix."""
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)


def rotation_log(matrix):
    """Recover the shortest axis-angle vector for nearby tool orientations."""
    angle = np.arccos(np.clip((np.trace(matrix)-1)/2, -1, 1))
    vector = np.array([matrix[2, 1]-matrix[1, 2], matrix[0, 2]-matrix[2, 0],
                       matrix[1, 0]-matrix[0, 1]])
    if angle < 1e-7:
        return vector / 2
    if np.pi-angle < 1e-5:
        raise RuntimeError("Tool orientation discontinuity near 180 degrees")
    return vector * angle / (2*np.sin(angle))


def sample_retreat(rng, config, reference=None):
    """Sample near a recorded retreat direction, or above an unprofiled anchor."""
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if reference.shape != (3,) or not np.isfinite(reference).all() or np.linalg.norm(reference) < 1e-8:
            raise RuntimeError("Invalid recorded retreat direction")
        axis = reference / np.linalg.norm(reference)
        tangent = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
        tangent /= np.linalg.norm(tangent)
        cosine = rng.uniform(np.cos(np.deg2rad(config.retreat_cone_deg)), 1)
        azimuth = rng.uniform(-np.pi, np.pi)
        direction = (cosine*axis + np.sqrt(1-cosine*cosine)*
                     (np.cos(azimuth)*tangent + np.sin(azimuth)*np.cross(axis, tangent)))
        return direction*rng.uniform(config.retreat_min_m, config.retreat_max_m)
    azimuth = rng.uniform(-np.pi, np.pi)
    z = rng.uniform(0.25, 1.0)
    direction = np.array([np.sqrt(1-z*z)*np.cos(azimuth),
                          np.sqrt(1-z*z)*np.sin(azimuth), z])
    return direction * rng.uniform(config.retreat_min_m, config.retreat_max_m)


def collect_episode(backend: PinBackend, seed: int, config=CollectionConfig()):
    """Return numeric trajectory arrays and a success/failure record.

    Each action maps observation i to observation i+1. Only return and close
    phases receive ``training_mask=True``; release/retreat are reset behavior,
    and lift/verify validate the grasp. Failed episodes always have a false mask.
    Images come from executed observations; no image sequence is reversed.
    The caller owns backend lifetime, including closing its camera resources.
    """
    if not np.isfinite(backend.hz) or backend.hz <= 0:
        raise ValueError("Backend control frequency must be finite and positive")
    observations, actions, phases = [], [], []
    result = dict(seed=int(seed), success=False, failure=None, reset=None)

    def observe():
        observation = {k: np.array(v, copy=True) for k, v in backend.observe().items()}
        if not observation or any(v.dtype.kind not in "biuf" or not np.isfinite(v).all()
                                  for v in observation.values()):
            raise RuntimeError("Backend returned non-numeric or non-finite observation")
        if observations and (observation.keys() != observations[0].keys() or
                             any(v.shape != observations[0][k].shape for k, v in observation.items())):
            raise RuntimeError("Backend observation schema changed within episode")
        observations.append(observation)
        return observation

    def step(position, rotation, opening, phase):
        backend.command(position.copy(), rotation.copy(), float(opening))
        observe()
        actions.append(np.r_[position, rotation.reshape(-1), opening])
        phases.append(PHASE_NAMES.index(phase))

    def hold(position, rotation, opening, seconds, phase):
        for _ in range(max(1, int(np.ceil(seconds*backend.hz)))):
            step(position, rotation, opening, phase)

    def move(start, end, rotation, opening, phase, end_rotation=None):
        delta = rotation_log(rotation.T @ end_rotation) if end_rotation is not None else np.zeros(3)
        duration = max(np.linalg.norm(end-start)/config.speed_m_s,
                       np.linalg.norm(delta)/np.deg2rad(config.angular_speed_deg_s))
        count = max(1, int(np.ceil(duration*backend.hz)))
        for fraction in np.linspace(0, 1, count+1)[1:]:
            step(start + fraction*(end-start), rotation @ rotation_exp(fraction*delta), opening, phase)

    try:
        result['reset'] = backend.reset(seed)
        initial = observe()
        if not backend.grasped():
            raise RuntimeError("Reset did not produce a verified grasp")
        anchor = initial['ee_position'].copy()
        rotation = initial['ee_rotation'].copy()
        if anchor.shape != (3,) or rotation.shape != (3, 3):
            raise RuntimeError("Expected EE position (3,) and rotation (3,3)")
        rng = np.random.default_rng(seed)
        reference = result['reset'].get('approach_retreat_direction')
        offset = sample_retreat(rng, config, reference)
        wrist_delta = np.deg2rad(rng.uniform(-config.rotation_jitter_deg, config.rotation_jitter_deg, 3))
        retreat_rotation = rotation @ rotation_exp(wrist_delta)
        result['retreat_offset_m'] = offset.tolist()
        result['retreat_rotation_vector_deg'] = np.rad2deg(wrist_delta).tolist()
        hold(anchor, rotation, backend.open_width, config.release_s, 'release')
        # Include the released anchor, then retrace measured EE positions. The
        # return camera stream is recorded while executing this path forward.
        retreat_start = len(observations)-1
        move(anchor, anchor+offset, rotation, backend.open_width, 'retreat', retreat_rotation)
        waypoints = [(o['ee_position'].copy(), o['ee_rotation'].copy()) for o in observations[retreat_start:]]
        position = observations[-1]['ee_position'].copy()
        current_rotation = observations[-1]['ee_rotation'].copy()
        for target, target_rotation in reversed(waypoints[:-1]):
            move(position, target, current_rotation, backend.open_width, 'return', target_rotation)
            position, current_rotation = target, target_rotation
        move(position, anchor, current_rotation, backend.open_width, 'return', rotation)
        hold(anchor, rotation, backend.closed_width, config.close_s, 'close')
        if not backend.grasped():
            raise RuntimeError("No bilateral grasp after closing")
        lifted = anchor + [0, 0, config.lift_m]
        move(anchor, lifted, rotation, backend.closed_width, 'lift')
        for _ in range(max(1, int(np.ceil(config.verify_s*backend.hz)))):
            step(lifted, rotation, backend.closed_width, 'verify')
            if not backend.verify_lift(config.lift_m*0.6):
                raise RuntimeError("Pin failed the sustained lift check")
        result['success'] = True
    except RuntimeError as exc:
        result['failure'] = str(exc)
    finally:
        backend.stop()

    arrays = {f'obs_{key}': np.stack([o[key] for o in observations])
              for key in observations[0]} if observations else {}
    arrays['actions'] = np.asarray(actions, dtype=np.float64).reshape(-1, 13)
    arrays['phases'] = np.asarray(phases, dtype=np.int8)
    arrays['phase_names'] = np.asarray(PHASE_NAMES)
    arrays['training_mask'] = (np.isin(arrays['phases'], [2, 3]) & result['success'])
    result['steps'] = len(actions)
    result['training_steps'] = int(arrays['training_mask'].sum())
    return arrays, result
