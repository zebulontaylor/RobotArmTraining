"""Collect uninterrupted physical regrasp cycles with one initial reset."""
from dataclasses import dataclass

import numpy as np

from collection.pin_regrasp import rotation_exp, rotation_log, sample_retreat

PHASE_NAMES = ('release', 'retreat', 'return', 'close', 'lift', 'verify', 'transfer', 'lower', 'settle')


@dataclass(frozen=True)
class ContinuousConfig:
    retreat_min_m: float = .03
    retreat_max_m: float = .09
    retreat_cone_deg: float = 12.
    rotation_jitter_deg: float = 6.
    speed_m_s: float = .035
    angular_speed_deg_s: float = 15.
    opening_speed_m_s: float = .04
    release_s: float = .7
    close_s: float = .35
    approach_close_start: float = .45
    approach_start_min_m: float = .012
    approach_start_max_m: float = .025
    approach_arrival_min_m: float = .0035
    approach_arrival_max_m: float = .006
    verify_s: float = .3
    lift_m: float = .04
    placement_radius_m: float = .025
    arc_m: float = .003

    def __post_init__(self):
        values = np.asarray(list(vars(self).values()))
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Continuous configuration values must be finite and positive')
        if self.retreat_min_m > self.retreat_max_m:
            raise ValueError('Retreat minimum exceeds maximum')
        if self.approach_close_start >= 1:
            raise ValueError('Approach closing must start before arrival')
        if not (self.approach_arrival_min_m <= self.approach_arrival_max_m <
                self.approach_start_min_m <= self.approach_start_max_m):
            raise ValueError('Expected ordered arrival and starting finger-opening ranges')


def collect_continuous(backend, seed, cycles=5, config=ContinuousConfig(), on_step=None,
                       store_images=False):
    """Reset once, then release, regrasp, lift and place using physical commands.

    The callback receives each executed observation, phase name and cycle index.
    Image arrays are streamed to the callback and omitted from storage by default.
    A failed cycle stops execution; successful earlier cycles retain their masks.
    """
    if not isinstance(cycles, int) or cycles < 1 or not np.isfinite(backend.hz) or backend.hz <= 0:
        raise ValueError('Expected positive cycles and control frequency')
    if not (backend.closed_width < config.approach_arrival_min_m and
            config.approach_start_max_m <= backend.open_width):
        raise ValueError('Approach opening ranges must lie within backend finger travel')
    rng = np.random.default_rng(seed)
    observations, actions, phases, cycle_ids = [], [], [], []
    result = dict(seed=int(seed), success=False, failure=None, reset=None, cycles=[])
    cycle = 0
    opening = backend.closed_width

    def observe(phase):
        raw = backend.observe()
        obs = {k: np.array(v, copy=True) for k, v in raw.items()
               if store_images or not k.startswith('image_')}
        if any(v.dtype.kind not in 'biuf' or not np.isfinite(v).all() for v in obs.values()):
            raise RuntimeError('Non-numeric or non-finite observation')
        if observations and (obs.keys() != observations[0].keys() or
                             any(v.shape != observations[0][k].shape for k, v in obs.items())):
            raise RuntimeError('Observation schema changed during continuous collection')
        if on_step is not None:
            on_step(raw, phase, cycle)
        observations.append(obs)
        return obs

    def step(p, r, width, phase):
        nonlocal opening
        backend.command(p.copy(), r.copy(), float(width))
        opening = float(width)
        observe(phase)
        actions.append(np.r_[p, r.ravel(), width])
        phases.append(PHASE_NAMES.index(phase))
        cycle_ids.append(cycle)
        if phase in ('lift', 'verify', 'transfer', 'lower', 'settle') and not backend.grasped():
            raise RuntimeError('Pin grasp lost during ' + phase)

    def move(p, target, r, target_r, width, phase, duration=0., arc=None, speed=None,
             closing_start=0.):
        delta = rotation_log(r.T @ target_r)
        # Quintic easing has peak derivative 1.875; reserve the same factor in
        # duration so commanded translation, rotation and fingers obey bounds.
        length = np.linalg.norm(target-p) + (np.pi*np.linalg.norm(arc) if arc is not None else 0.)
        seconds = max(duration, 1.875*length/(speed or config.speed_m_s),
                      1.875*np.linalg.norm(delta)/np.deg2rad(config.angular_speed_deg_s),
                      1.875*abs(width-opening)/(config.opening_speed_m_s*(1-closing_start)))
        count = max(1, int(np.ceil(seconds*backend.hz)))
        start_width = opening
        for t in np.linspace(0., 1., count+1)[1:]:
            u = t*t*t*(10+t*(-15+6*t))
            point = p + u*(target-p)
            if arc is not None:
                point = point + np.sin(np.pi*u)*arc
            grip_t = np.clip((t-closing_start)/(1-closing_start), 0., 1.)
            grip_u = grip_t**3*(10+grip_t*(-15+6*grip_t))
            step(point, r @ rotation_exp(u*delta), start_width+grip_u*(width-start_width), phase)

    try:
        result['reset'] = backend.reset(seed)
        obs = observe('initial')
        anchor, rotation = obs['ee_position'].copy(), obs['ee_rotation'].copy()
        center = anchor.copy()
        initial_rotation = rotation.copy()
        reference = result['reset'].get('approach_retreat_direction')
        if not backend.grasped():
            raise RuntimeError('Initial pin grasp failed')
        for cycle in range(cycles):
            record = dict(cycle=cycle, success=False)
            result['cycles'].append(record)
            if not backend.supported():
                raise RuntimeError('Pin is not table-supported before release')
            move(anchor, anchor, rotation, rotation, backend.open_width, 'release', config.release_s)
            offset = sample_retreat(rng, config, reference)
            retreat_r = rotation @ rotation_exp(np.deg2rad(rng.uniform(-config.rotation_jitter_deg, config.rotation_jitter_deg, 3)))
            arc = rng.uniform(-config.arc_m, config.arc_m, 3); arc[2] = 0
            speed = config.speed_m_s*rng.uniform(.65, 1.)
            start_opening = rng.uniform(config.approach_start_min_m, config.approach_start_max_m)
            arrival_opening = rng.uniform(config.approach_arrival_min_m, config.approach_arrival_max_m)
            record.update(approach_start_opening_m=float(start_opening),
                          approach_arrival_opening_m=float(arrival_opening))
            move(anchor, anchor+offset, rotation, retreat_r, start_opening, 'retreat', arc=arc, speed=speed)
            # Narrow the fingers during the final approach, leaving clearance
            # around the pin until the tool reaches the grasp position.
            move(anchor+offset, anchor, retreat_r, rotation, arrival_opening,
                 'return', arc=-arc, speed=speed, closing_start=config.approach_close_start)
            move(anchor, anchor, rotation, rotation, backend.closed_width, 'close', config.close_s)
            if not backend.grasped():
                raise RuntimeError('No bilateral pin grasp after closing')
            lifted = anchor + [0, 0, config.lift_m]
            move(anchor, lifted, rotation, rotation, backend.closed_width, 'lift')
            for _ in range(max(1, int(np.ceil(config.verify_s*backend.hz)))):
                step(lifted, rotation, backend.closed_width, 'verify')
                if not backend.verify_lift(config.lift_m*.6):
                    raise RuntimeError('Pin failed sustained lift verification')
            target = center.copy(); target[:2] += rng.uniform(-config.placement_radius_m, config.placement_radius_m, 2)
            high = target + [0, 0, config.lift_m]
            placement_rotation = rotation_exp(np.array([0., 0., np.deg2rad(rng.uniform(
                -config.rotation_jitter_deg, config.rotation_jitter_deg))])) @ initial_rotation
            move(lifted, high, rotation, placement_rotation, backend.closed_width, 'transfer', arc=arc, speed=speed)
            rotation = placement_rotation
            move(high, target, rotation, rotation, backend.closed_width, 'lower')
            move(target, target, rotation, rotation, backend.closed_width, 'settle', .5)
            if not backend.supported():
                raise RuntimeError('Placed pin is not table-supported')
            pose = observations[-1].get('object_pose')
            if pose is not None and pose.shape == (7,) and 1-2*(pose[4]**2+pose[5]**2) < np.cos(np.deg2rad(10)):
                raise RuntimeError('Placed pin is not upright')
            record.update(success=True, retreat_offset_m=offset.tolist(),
                          retreat_rotation=retreat_r.tolist(), placement_position=target.tolist(),
                          placement_rotation=rotation.tolist(), speed_m_s=float(speed))
            anchor = target
        result['success'] = True
    except RuntimeError as exc:
        result['failure'] = str(exc)
    finally:
        backend.stop()
    arrays = {f'obs_{key}': np.stack([o[key] for o in observations]) for key in observations[0]} if observations else {}
    arrays.update(actions=np.asarray(actions).reshape(-1, 13), phases=np.asarray(phases, dtype=np.int8),
                  phase_names=np.asarray(PHASE_NAMES), cycle_ids=np.asarray(cycle_ids, dtype=np.int32))
    successful = [r['cycle'] for r in result['cycles'] if r['success']]
    arrays['training_mask'] = np.isin(arrays['cycle_ids'], successful) & np.isin(arrays['phases'], [2, 3])
    result.update(steps=len(actions), completed_cycles=len(successful), training_steps=int(arrays['training_mask'].sum()))
    return arrays, result
