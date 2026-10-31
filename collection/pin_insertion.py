"""Collect pickup-style held-pin approaches ending above the carrier hole."""
from dataclasses import dataclass

import numpy as np

from collection.pin_regrasp import NumericTrajectory, rotation_exp, rotation_log, sample_retreat

PHASE_NAMES = ('retreat', 'carrier_drag', 'pin_relocate', 'return', 'verify')
TRAINING_PHASES = ('return',)


@dataclass(frozen=True)
class InsertionCollectionConfig:
    """Alignment motion settings; the class name preserves existing imports."""

    cycles: int = 3
    endpoint_clearance_m: float = .001
    retreat_min_m: float = .02
    retreat_max_m: float = .03
    retreat_cone_deg: float = 4.
    speed_m_s: float = .02
    arc_m: float = .002
    verify_s: float = .25
    scene_xy_m: float = .03
    carrier_drag_m: float = .012
    carrier_drag_speed_m_s: float = .005
    max_steps: int = 1400

    def __post_init__(self):
        values = [self.endpoint_clearance_m, self.retreat_min_m, self.retreat_max_m,
                  self.retreat_cone_deg, self.speed_m_s, self.arc_m, self.verify_s, self.scene_xy_m,
                  self.carrier_drag_m, self.carrier_drag_speed_m_s]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('Alignment settings must be finite and nonnegative')
        if not .0007 <= self.endpoint_clearance_m <= .0013:
            raise ValueError('Endpoint clearance must be between .0007 and .0013 m')
        if not .02 <= self.retreat_min_m <= self.retreat_max_m <= .04:
            raise ValueError('Retreat distances must be ordered within [.02, .04] m')
        if not 0 < self.speed_m_s <= .025 or self.verify_s <= 0:
            raise ValueError('Speed must be in (0, .025] m/s and verify_s positive')
        if self.retreat_cone_deg > 8 or self.arc_m > .003 or self.scene_xy_m > .03:
            raise ValueError('Retreat cone must be <= 8 degrees, arc <= .003 m, and shared scene XY <= .03 m')
        if not .004 <= self.carrier_drag_m <= .02 or not 0 < self.carrier_drag_speed_m_s <= .005:
            raise ValueError('Carrier drag must be in [.004, .02] m and speed in (0, .005] m/s')
        if any(not isinstance(v, int) or v < 1 for v in (self.cycles, self.max_steps)):
            raise ValueError('cycles and max_steps must be positive integers')


def collect_insertion_episode(backend, seed, config=InsertionCollectionConfig()):
    """Approach obliquely toward the hole with fixed wrist and grippers.

    Reset prepares a held pin near the 1 mm hover anchor. Setup aligns it and
    physically retreats toward the arm base at 40 degrees before recording.
    Return frames come from executing the measured retreat path in reverse
    waypoint order. Every measured pin tip must remain above the carrier mouth.
    """
    trajectory = NumericTrajectory(PHASE_NAMES)
    cycle_ids = []
    result = dict(seed=int(seed), success=False, failure=None, reset=None,
                  task='alignment', alignment_mode='alignment_only',
                  approach_style='pickup_toward_base_40deg', setup_steps=0,
                  requested_cycles=config.cycles, completed_cycles=0, cycles=[])
    env = backend.env
    cycle = 0
    recording = False
    frozen_carrier = None
    max_carrier_drift = 0.

    def tool_pose():
        return (backend.arm.ee_pos().copy(),
                backend.sim.data.site_xmat[backend.arm.ee_site].reshape(3, 3).copy())

    def execute(action, phase):
        nonlocal max_carrier_drift
        if len(trajectory.actions) + result['setup_steps'] >= config.max_steps:
            raise RuntimeError('Collection step limit reached')
        _, _, terminated, truncated, info = env.step(action)
        if recording:
            trajectory.record(backend, action, phase)
            cycle_ids.append(cycle)
        else:
            result['setup_steps'] += 1
        result['validation'] = info
        status = backend.alignment_status()
        result['alignment'] = status
        if not status['carrier_table_supported']:
            raise RuntimeError('Carrier lost loaded table support')
        if phase in ('return', 'verify') and frozen_carrier is not None:
            position, orientation = frozen_carrier
            drift = float(np.linalg.norm(backend.sim.data.xpos[backend._carrier]-position))
            turn = np.linalg.norm(rotation_log(backend.sim.data.xmat[backend._carrier].reshape(3, 3) @ orientation.T))
            max_carrier_drift = max(max_carrier_drift, drift)
            if drift > .0005 or turn > np.deg2rad(.5):
                raise RuntimeError('Carrier moved during the pin approach')
        if status['clearance_m'] <= 0:
            raise RuntimeError('Pin tip crossed the carrier mouth during alignment')
        if info['failure'] or truncated:
            raise RuntimeError(info['failure'] or 'Alignment step limit reached')
        if terminated:
            raise RuntimeError('Unexpected insertion during alignment-only collection')
        if not backend.grasped() or not backend._holder_grasped():
            raise RuntimeError('Lost loaded pin or carrier grasp')
        return status

    def pose_action(position, orientation, speed=None):
        p, r = env.targets['left']
        action = np.zeros(14)
        delta = (position-p)/.001
        limit = (speed or config.speed_m_s)/backend.hz/.001
        action[:3] = delta * min(1., limit/max(np.linalg.norm(delta), 1e-12))
        action[3:6] = np.clip(rotation_log(orientation @ r.T)/np.deg2rad(.5), -.2, .2)
        return np.clip(action, -1., 1.)

    def anchor_control():
        s = backend.sim
        carrier_r = s.data.xmat[backend._carrier].reshape(3, 3)
        pin_r = s.data.xmat[backend._pin].reshape(3, 3)
        axis_cross = np.cross(pin_r[:, 2], carrier_r[:, 2])
        angle = np.arctan2(np.linalg.norm(axis_cross), pin_r[:, 2] @ carrier_r[:, 2])
        correction = rotation_exp(axis_cross * angle / max(np.linalg.norm(axis_cross), 1e-12))
        goal = backend.mouth_position() + carrier_r[:, 2]*config.endpoint_clearance_m
        position = goal + correction @ (backend.arm.ee_pos()-s.data.xpos[backend._pin])
        orientation = correction @ tool_pose()[1]
        return pose_action(position, orientation, speed=.004)

    def aligned(status):
        return (.0007 <= status['clearance_m'] <= .0013
                and status['mouth_error_m'] < .00017 and status['tilt_deg'] < 3)

    def verify_anchor():
        samples = []
        for _ in range(verify_steps):
            status = execute(anchor_control(), 'verify')
            if not aligned(status):
                raise RuntimeError('Pin failed sustained alignment above the carrier mouth')
            samples.append(status)
        return samples

    def retreat(anchor_p, anchor_r, offset):
        waypoints = [tool_pose()]
        # This is the pickup collector's eased curved motion, with the same
        # 1.875 peak-derivative factor used to bound the commanded speed.
        length = np.linalg.norm(offset) + np.pi*config.arc_m
        count = max(1, int(np.ceil(1.875*length/config.speed_m_s*backend.hz)))
        for t in np.linspace(0, 1, count+1)[1:]:
            u = t*t*t*(10+t*(-15+6*t))
            point = anchor_p + u*offset + [0, 0, config.arc_m*np.sin(np.pi*u)]
            execute(pose_action(point, anchor_r), 'retreat')
            waypoints.append(tool_pose())
        for _ in range(verify_steps):
            execute(pose_action(anchor_p+offset, anchor_r), 'retreat')
            waypoints.append(tool_pose())
        return waypoints

    def reposition_carrier(target_xy):
        before = backend.sim.data.xpos[backend._carrier].copy()
        right_start = env.targets['right'][0].copy()
        right_goal = right_start.copy()
        right_goal[:2] += target_xy-before[:2]
        length = np.linalg.norm(right_goal-right_start)
        count = max(1, int(np.ceil(1.875*length/config.carrier_drag_speed_m_s*backend.hz)))
        for t in np.linspace(0, 1, count+1)[1:]:
            u = t*t*t*(10+t*(-15+6*t))
            goal = right_start + u*(right_goal-right_start)
            action = np.zeros(14)
            action[7:10] = np.clip((goal-env.targets['right'][0])/.00025, -1., 1.)
            execute(action, 'carrier_drag')
            if abs(backend.sim.data.xpos[backend._carrier, 2]-before[2]) > .001:
                raise RuntimeError('Carrier lifted during planar drag')
        positions = []
        for _ in range(max(verify_steps, 10)):
            execute(np.zeros(14), 'carrier_drag')
            positions.append(backend.sim.data.xpos[backend._carrier].copy())
        after = positions[-1]
        displacement = after-before
        if np.linalg.norm(displacement[:2]) < max(.003, config.carrier_drag_m*.75):
            raise RuntimeError('Carrier drag did not produce the required measured motion')
        if np.linalg.norm(after[:2]-target_xy) > .001:
            raise RuntimeError('Carrier did not reach its requested drag position')
        if abs(displacement[2]) > .001 or np.max(np.linalg.norm(np.asarray(positions)-after, axis=1)) > .0003:
            raise RuntimeError('Carrier did not settle on the table after dragging')
        return displacement, dict(before_position_m=before.tolist(), after_position_m=after.tolist(),
                                  displacement_m=displacement.tolist(), target_xy_m=target_xy.tolist())

    def relocate_pin(target, orientation):
        start = tool_pose()[0]
        count = max(1, int(np.ceil(1.875*np.linalg.norm(target-start)/config.speed_m_s*backend.hz)))
        for t in np.linspace(0, 1, count+1)[1:]:
            u = t*t*t*(10+t*(-15+6*t))
            execute(pose_action(start+u*(target-start), orientation), 'pin_relocate')
        for _ in range(verify_steps):
            execute(pose_action(target, orientation), 'pin_relocate')

    try:
        result['reset'] = backend.reset(seed)
        result['reset']['wrist_clearance_yaw_deg'] = -12
        if backend.alignment_status()['clearance_m'] <= 0:
            raise RuntimeError('Reset pin tip is not above the carrier mouth')
        rng = np.random.default_rng(seed)
        original_carrier = backend.sim.data.xpos[backend._carrier].copy()
        holder_forward = env.targets['right'][1][:, 0].copy()
        holder_forward[2] = 0
        holder_forward /= np.linalg.norm(holder_forward)
        verify_steps = max(1, int(np.ceil(config.verify_s*backend.hz)))
        for _ in range(100):
            if aligned(execute(anchor_control(), 'verify')):
                break
        else:
            raise RuntimeError('Could not establish held-pin hover anchor during setup')
        verify_anchor()
        anchor_p, anchor_r = tool_pose()
        base = backend.sim.data.xpos[backend.sim.model.body('left_base').id]
        toward_base = base-anchor_p
        toward_base[2] = 0
        toward_base /= np.linalg.norm(toward_base)
        axis = np.cos(np.deg2rad(40))*toward_base + [0, 0, np.sin(np.deg2rad(40))]
        result['approach_retreat_direction'] = axis.tolist()
        offset = sample_retreat(rng, config, axis)
        waypoints = retreat(anchor_p, anchor_r, offset)
        result['setup_time_s'] = result['setup_steps']/backend.hz
        result['setup_anchor_position_m'] = anchor_p.tolist()
        result['initial_retreat_offset_m'] = offset.tolist()
        result['initial_approach_start_position_m'] = tool_pose()[0].tolist()
        recording = True
        trajectory.observe(backend)
        for cycle in range(config.cycles):
            start = len(trajectory.actions)
            drag_record = None
            if cycle:
                anchor_p, anchor_r = tool_pose()
                offset = sample_retreat(rng, config, axis)
                waypoints = retreat(anchor_p, anchor_r, offset)
                target = original_carrier[:2] + holder_forward[:2]*config.carrier_drag_m*(1 if cycle % 2 else -1)
                displacement, drag_record = reposition_carrier(target)
                anchor_p += displacement
                waypoints = [(p+displacement, r) for p, r in waypoints]
                relocate_pin(waypoints[-1][0], waypoints[-1][1])
            frozen_carrier = (backend.sim.data.xpos[backend._carrier].copy(),
                              backend.sim.data.xmat[backend._carrier].reshape(3, 3).copy())
            max_carrier_drift = 0.
            retreat_status = backend.alignment_status()
            actual_offset = tool_pose()[0]-anchor_p
            if (np.linalg.norm(actual_offset[:2]) < .01
                    or actual_offset[:2] @ toward_base[:2] < .01
                    or actual_offset[2] < .005 or retreat_status['clearance_m'] <= .005):
                raise RuntimeError('Physical retreat lacks the required oblique approach clearance')
            return_start = len(trajectory.actions)
            for position, orientation in reversed(waypoints[:-1]):
                execute(pose_action(position, orientation), 'return')
            for _ in range(100):
                status = execute(anchor_control(), 'return')
                if aligned(status):
                    break
            else:
                raise RuntimeError('Pin did not recover hover alignment after the approach')
            endpoint = verify_anchor()
            result['cycles'].append(dict(cycle=cycle, action_start=start,
                return_action_start=return_start, action_end=len(trajectory.actions),
                anchor_position_m=anchor_p.tolist(), retreat_offset_m=offset.tolist(),
                actual_retreat_offset_m=actual_offset.tolist(),
                retreat=retreat_status, endpoint_samples=endpoint, carrier_drag=drag_record,
                carrier_reference_position_m=frozen_carrier[0].tolist(),
                carrier_max_approach_drift_m=max_carrier_drift))
            result['completed_cycles'] += 1
        result['success'] = True
    except RuntimeError as exc:
        result['failure'] = str(exc)
        if result['reset'] is not None and not trajectory.observations:
            trajectory.observe(backend)
    finally:
        backend.stop()
    arrays = trajectory.arrays(14, TRAINING_PHASES, result['success'])
    arrays['cycle_ids'] = np.asarray(cycle_ids, dtype=np.int32)
    result['steps'] = len(arrays['actions'])
    result['training_steps'] = int(arrays['training_mask'].sum())
    return arrays, result


AlignmentCollectionConfig = InsertionCollectionConfig
collect_alignment_episode = collect_insertion_episode
