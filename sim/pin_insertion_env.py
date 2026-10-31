"""Two-arm, contact-driven pin insertion starting from an already held pin."""
from dataclasses import dataclass
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from sim.actuator_env import ASSETS, TABLE_Z
from sim.bimanual_actuator_env import BimanualActuatorSim
from sim.panthera_env import mat_to_quat
from tools.try_actuator_assembly import pin_placement, pin_target_pose


def rotation(vector):
    """Rotation matrix from a rotation vector in radians."""
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, vector / angle, angle)
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, quat)
    return result.reshape(3, 3)


@dataclass(frozen=True)
class PinInsertionConfig:
    start_xy_m: float = .005
    start_height_m: tuple = (.020, .040)
    pin_grip_translation_m: float = .001
    pin_grip_rotation_deg: float = 1.
    holder_translation_m: float = .0005
    holder_rotation_deg: float = 2.
    carrier_center_xy_m: tuple = (.26, 0.)
    carrier_xy_m: float = .030
    carrier_yaw_deg: float = 10.
    max_steps: int = 400
    success_steps: int = 5
    contact_limit_n: float = 35.
    reset_attempts: int = 30
    reward_version: str = 'legacy'
    carrier_on_table: bool = False
    holder_grasp_height_m: float = .004
    holder_workspace_m: float = .002
    carrier_travel_limit_m: float = .002

    def __post_init__(self):
        values = (self.start_xy_m, self.pin_grip_translation_m,
                  self.pin_grip_rotation_deg, self.holder_translation_m,
                  self.holder_rotation_deg, self.carrier_xy_m, self.carrier_yaw_deg)
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('Randomization widths must be finite and nonnegative')
        if len(self.carrier_center_xy_m) != 2 or not np.isfinite(self.carrier_center_xy_m).all():
            raise ValueError('Carrier center must contain two finite coordinates')
        if (len(self.start_height_m) != 2 or not np.isfinite(self.start_height_m).all()
                or not 0 < self.start_height_m[0] <= self.start_height_m[1]):
            raise ValueError('Start heights must be positive and ordered')
        for value in (self.max_steps, self.success_steps, self.reset_attempts):
            if not isinstance(value, int) or value <= 0:
                raise ValueError('Step counts and reset attempts must be positive integers')
        if not np.isfinite(self.contact_limit_n) or self.contact_limit_n <= 0:
            raise ValueError('Contact limit must be positive and finite')
        if not isinstance(self.carrier_on_table, bool):
            raise ValueError('carrier_on_table must be boolean')
        if (not np.isfinite([self.holder_grasp_height_m, self.holder_workspace_m, self.carrier_travel_limit_m]).all()
                or not 0 < self.holder_grasp_height_m <= .02
                or not 0 < self.holder_workspace_m <= .03
                or not 0 < self.carrier_travel_limit_m <= .03):
            raise ValueError('Holder height must be in (0, .02] m; workspace and carrier travel limits in (0, .03] m')
        if self.reward_version not in ('legacy', 'alignment_v1', 'alignment_v2'):
            raise ValueError('Reward version must be legacy, alignment_v1 or alignment_v2')


class BimanualPinInsertionEnv(gym.Env):
    """Left arm inserts pin 2; right arm stabilizes the table-supported carrier.

    Each arm takes seven normalized incremental actions: world XYZ, world
    rotation vector, and gripper opening. Observations use privileged geometry
    in metres/radians. Part poses are assigned only during reset; subsequent
    motion uses the original servos, friction and collision geometry.
    """
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 20}

    def __init__(self, config=None, render_mode=None, profile_path=None, *, setup_profile=None):
        super().__init__()
        self.config = config or PinInsertionConfig()
        if render_mode not in (None, *self.metadata['render_modes']):
            raise ValueError(f'Unsupported render mode: {render_mode}')
        self.render_mode = render_mode
        self.sim = BimanualActuatorSim(layout='side_by_side', setup_profile=setup_profile)
        self.model, self.data = self.sim.model, self.sim.data
        for arm in self.sim.arms.values():
            self.model.site_rgba[arm.ee_site, 3] = 0
        self.profile = json.loads(Path(profile_path or ASSETS / 'pin_reset_profile.json').read_text())
        if not self.profile.get('samples'):
            raise ValueError('Reset profile must contain recorded arm orientations')
        self.pin_name = 'pin_2'
        self.pin_id = self.model.body(self.pin_name).id
        self.gear_id = self.model.body('gear_3').id
        self.carrier_id = self.model.body('large_carrier').id
        self.action_space = spaces.Box(-1., 1., (14,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'proprioception': spaces.Box(-np.inf, np.inf, (28,), dtype=np.float32),
            'task': spaces.Box(-np.inf, np.inf, (33,), dtype=np.float32),
            'controller': spaces.Box(-np.inf, np.inf, (64,), dtype=np.float32),
        })
        self._renderer = self._viewer = None
        self._ready = False
        self._finger_bodies = {self.model.body(n).id for n in ('L_finger', 'R_finger')}
        self._holder_fingers = {self.model.body(n).id for n in ('right_L_finger', 'right_R_finger')}

    def _aligned_rotation(self, sample, side):
        measured = np.array(sample[side]['rotation_base'])
        base = self.model.body(side + '_base').id
        return self.data.xmat[base].reshape(3, 3) @ measured

    def _place_arm(self, side, position, orientation, opening, sample):
        arm = self.sim.arms[side]
        starts = [np.array(sample[side]['joints_rad']), arm._q_nominal]
        starts += [self.np_random.uniform(*arm.arm_range.T) for _ in range(4)]
        for start in starts:
            q, pe, re = arm.ik(position, mat_to_quat(orientation), q_init=start,
                               iters=160, max_joint_step=None, posture_gain=0,
                               min_damping=.001, pos_tol=.00003)
            if pe < .00015 and re < .002:
                self.data.qpos[arm.arm_qadr] = q
                self.data.qpos[arm.finger_qadr] = [opening, -opening]
                arm.set_arm_ctrl(q, immediate=True)
                arm.set_gripper(0)
                mujoco.mj_forward(self.model, self.data)
                return True
        return False

    def _pinched(self):
        return self._body_pinched(self.pin_id, self._finger_bodies)

    def _body_pinched(self, body_id, fingers):
        touched = set()
        for contact in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if body_id in (b1, b2):
                touched.add(b2 if b1 == body_id else b1)
        return fingers <= touched

    def _contact_load(self):
        body1 = self.model.geom_bodyid[self.data.contact.geom1]
        body2 = self.model.geom_bodyid[self.data.contact.geom2]
        finger1, finger2 = self._finger_bodies
        selected = ((body1 == self.pin_id) | (body2 == self.pin_id))
        selected &= ((body1 != finger1) & (body1 != finger2)
                     & (body2 != finger1) & (body2 != finger2))
        load = 0.
        force = np.empty(6)
        for index in np.flatnonzero(selected):
            mujoco.mj_contactForce(self.model, self.data, int(index), force)
            load += max(0., force[0])
        return load

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ready = False
        randomize = (options or {}).get('randomize', True)
        cfg = self.config
        for attempt in range(cfg.reset_attempts):
            self.sim.reset(randomize=False)
            carrier_offset = np.zeros(3)
            carrier_yaw = 0.
            if randomize:
                carrier_offset[:2] = self.np_random.uniform(-cfg.carrier_xy_m, cfg.carrier_xy_m, 2)
                carrier_yaw = self.np_random.uniform(-cfg.carrier_yaw_deg, cfg.carrier_yaw_deg)
            scene_rotation = rotation([0, 0, np.deg2rad(carrier_yaw)])
            # Present the target hole toward the left base, leaving the
            # opposite side of the carrier clear for the stabilizing grasp.
            fixture = self.model.body('small_carrier').id
            self.model.body_pos[fixture, :2] = np.asarray(cfg.carrier_center_xy_m) + carrier_offset[:2]
            self.model.geom_pos[self.model.geom('fixture').id, :2] = self.model.body_pos[fixture, :2]
            self.model.body_quat[fixture] = mat_to_quat(
                scene_rotation @ rotation([0, 0, -np.pi / 2]))
            mujoco.mj_forward(self.model, self.data)
            # In insertion scenes only the target gear is seated. Bare-table
            # alignment leaves it at pickup so the carrier can slide freely.
            for name in ('large_carrier', 'gear_3'):
                position, orientation = self.sim.assembled_pose(name)
                if cfg.carrier_on_table:
                    if name == 'large_carrier':
                        position[2] = TABLE_Z + .0002
                    else:
                        position = np.r_[self.sim.stations[name], TABLE_Z + .0002]
                self.sim.set_part_pose(name, position, orientation)
            if cfg.carrier_on_table:
                # Alignment can drag the held plate on bare table instead of
                # trying to pull it sideways over the fixed assembly spindle.
                self.model.body_pos[fixture, :2] = [.54, .38]
                self.model.geom_pos[self.model.geom('fixture').id, :2] = [.54, .38]
            mujoco.mj_forward(self.model, self.data)
            index = int(self.np_random.integers(len(self.profile['samples']))) if randomize else 0
            sample = self.profile['samples'][index]
            holder_r = scene_rotation @ self._aligned_rotation(sample, 'right')
            holder_shift = np.zeros(3)
            holder_angles = np.zeros(3)
            grip_shift = np.zeros(3)
            grip_angles = np.zeros(3)
            offset = np.array([0., 0., np.mean(cfg.start_height_m)])
            if randomize:
                holder_shift[[0, 2]] = self.np_random.uniform(-cfg.holder_translation_m, cfg.holder_translation_m, 2)
                holder_angles = self.np_random.uniform(-cfg.holder_rotation_deg, cfg.holder_rotation_deg, 3)
                grip_shift[[0, 2]] = self.np_random.uniform(-cfg.pin_grip_translation_m, cfg.pin_grip_translation_m, 2)
                grip_angles = self.np_random.uniform(-cfg.pin_grip_rotation_deg, cfg.pin_grip_rotation_deg, 3)
                offset[:2] = self.np_random.uniform(-cfg.start_xy_m, cfg.start_xy_m, 2)
                offset[2] = self.np_random.uniform(*cfg.start_height_m)
            holder_r = holder_r @ rotation(np.deg2rad(holder_angles))
            horizontal = holder_r[:, 0].copy()
            horizontal[2] = 0
            horizontal /= np.linalg.norm(horizontal)
            pitch = np.arcsin(np.clip(-holder_r[2, 0], -1, 1))
            holder_p = (self.data.xpos[self.carrier_id] + horizontal * .010 * np.sin(pitch)
                        + [0, 0, cfg.holder_grasp_height_m] + holder_r @ holder_shift)
            if not self._place_arm('right', holder_p, holder_r, .027, sample):
                continue
            target, _ = pin_target_pose(self.sim, self.pin_name)
            # Height is measured from the upper hole mouth to the pin bottom.
            mouth = target.copy()
            mouth[2] = self.data.xpos[self.carrier_id, 2] + .01725
            pin_bottom = mouth + offset
            left_r = scene_rotation @ self._aligned_rotation(sample, 'left')
            left_p = pin_bottom + [0, 0, .013]
            if not self._place_arm('left', left_p, left_r, .0045, sample):
                continue
            pin_r = left_r @ rotation(np.deg2rad(grip_angles)) @ left_r.T
            self.sim.set_part_pose(self.pin_name, pin_bottom + left_r @ grip_shift,
                                   mat_to_quat(pin_r))
            mujoco.mj_forward(self.model, self.data)
            self.carrier_reference = self.data.xpos[self.carrier_id].copy()
            self.sim.step(200)
            if (not self._pinched() or not self._body_pinched(self.carrier_id, self._holder_fingers) or
                    np.linalg.norm(self.data.xpos[self.carrier_id] - self.carrier_reference) > .002):
                continue
            # Reject unwanted reset penetrations instead of starting training
            # with impulses from an arm passing through a part or the table.
            if any(c.dist < -.0005 for c in self.data.contact):
                continue
            self.reset_parameters = dict(
                profile_episode=sample['episode_id'], profile_time_s=sample['time_s'],
                profile_index=index, attempts=attempt + 1,
                arm_offset_m=offset.copy(), pin_grip_translation_m=grip_shift.copy(),
                pin_grip_rotation_deg=grip_angles.copy(),
                holder_translation_m=holder_shift.copy(), holder_rotation_deg=holder_angles.copy(),
                carrier_offset_m=carrier_offset.copy(), carrier_yaw_deg=float(carrier_yaw),
                carrier_on_table=cfg.carrier_on_table, holder_grasp_height_m=cfg.holder_grasp_height_m)
            self.targets = {side: (arm.ee_pos().copy(),
                                  self.data.site_xmat[arm.ee_site].reshape(3, 3).copy())
                            for side, arm in self.sim.arms.items()}
            self.origins = {s: (p.copy(), r.copy()) for s, (p, r) in self.targets.items()}
            self.commands = {s: self.data.ctrl[a.act_ids].copy()
                             for s, a in self.sim.arms.items()}
            self.velocity = {s: np.zeros(6) for s in self.sim.arms}
            # Retain the settled servo preload when control starts; replacing
            # it with measured joints would let the tools sag under gravity.
            self.bias = {s: np.clip(self.commands[s] - a.q, -.012, .012)
                         for s, a in self.sim.arms.items()}
            self.grips = {s: 0. for s in self.sim.arms}
            self.steps = self.seated_steps = 0
            self.peak_load = self._contact_load()
            self._ready = True
            info = self._info()
            self._potential = self._reward_score(info)
            self.reset_parameters['actual_pin_in_grip_m'] = self._in_hand()[0].copy()
            if self.render_mode == 'human':
                self.render()
            return self._observation(info), dict(info, reset=self.reset_parameters)
        raise RuntimeError(f'No reachable, pinched reset after {cfg.reset_attempts} attempts')

    def _in_hand(self):
        arm = self.sim.arms['left']
        r = self.data.site_xmat[arm.ee_site].reshape(3, 3)
        return (r.T @ (self.data.xpos[self.pin_id] - arm.ee_pos()),
                r.T @ self.data.xmat[self.pin_id].reshape(3, 3))

    def _info(self):
        placement = pin_placement(self.sim, self.pin_name)
        target, _ = pin_target_pose(self.sim, self.pin_name)
        delta = self.data.xpos[self.pin_id] - target
        carrier_error = np.linalg.norm(self.data.xpos[self.carrier_id] - self.carrier_reference)
        carrier_axis = self.data.xmat[self.carrier_id].reshape(3, 3)[:, 2]
        supported = carrier_error < self.config.carrier_travel_limit_m and carrier_axis[2] > np.cos(np.deg2rad(3))
        holder_pinched = self._body_pinched(self.carrier_id, self._holder_fingers)
        return dict(placement, is_success=bool(placement['success'] and supported and holder_pinched),
                    pin_delta_m=delta, carrier_error_m=float(carrier_error),
                    pin_contact_load_n=float(self.peak_load), pinched=self._pinched(),
                    holder_pinched=holder_pinched,
                    carrier_supported=bool(supported))

    @staticmethod
    def _score(info):
        d = info['pin_delta_m']
        return float(-80 * np.linalg.norm(d[:2]) - 20 * abs(d[2])
                     - .02 * info['tilt_deg'] - 50 * info['carrier_error_m'])

    @staticmethod
    def _alignment_components(info):
        """Score the shaft axis at both bore ends, even before it enters.

        Between one and two times the success alignment tolerances, axial
        progress fades out. The 80 mm depth anchor exceeds the reset height;
        aligning therefore improves the potential rather than introducing a
        new depth penalty. An axis pointing away from a bore has no gate.
        """
        errors = [info['carrier_bore_error_mm'], info['gear_bore_error_mm']]
        usable = all(value is not None and np.isfinite(value) for value in errors)
        carrier, gear = [float(value) / 1000 if value is not None and np.isfinite(value)
                         else .010 for value in errors]
        ratio = max(carrier / .00017, gear / .00015, info['tilt_deg'] / 3.)
        gate = float(np.clip(2. - ratio, 0., 1.)) if usable else 0.
        depth = abs(info['axial_error_mm']) / 1000
        effective_depth = depth + (1. - gate) * max(.080 - depth, 0.)
        return dict(bore_alignment=-100. * (carrier + gear),
                    axial=-20. * effective_depth, tilt=-.02 * info['tilt_deg'],
                    carrier=-50. * info['carrier_error_m'], alignment_gate=gate)

    @staticmethod
    def _alignment_v2_components(info):
        """Bound alignment potential when a nearly horizontal axis misses a bore.

        Distances retain their near-zero slope and saturate smoothly. Invalid
        bore intersections receive the full saturated cost, so flipping the
        shaft past horizontal cannot improve its alignment potential. The
        resulting potential lies in [-7.7, 0]; contact costs remain separate.
        """
        def bounded_distance(value, scale):
            if value is None or not np.isfinite(value):
                return float(scale)
            return float(scale * np.tanh(abs(float(value)) / scale))

        errors = [info['carrier_bore_error_mm'], info['gear_bore_error_mm']]
        usable = all(value is not None and np.isfinite(value) for value in errors)
        raw = [abs(float(value)) / 1000 if value is not None and np.isfinite(value)
               else np.inf for value in errors]
        tilt = info['tilt_deg']
        tilt = float(np.clip(tilt, 0., 180.)) if np.isfinite(tilt) else 180.
        ratio = max(raw[0] / .00017, raw[1] / .00015, tilt / 3.)
        gate = float(np.clip(2. - ratio, 0., 1.)) if usable else 0.
        depth = abs(info['axial_error_mm']) / 1000
        effective_depth = .080 - gate * (.080 - depth) if depth < .080 else depth
        bore = [bounded_distance(value, .010) for value in raw]
        return dict(bore_alignment=-100. * sum(bore),
                    axial=-20. * bounded_distance(effective_depth, .080),
                    tilt=-.02 * tilt,
                    carrier=-50. * bounded_distance(info['carrier_error_m'], .010),
                    alignment_gate=gate)

    def _reward_score(self, info):
        if self.config.reward_version == 'legacy':
            return self._score(info)
        components = (self._alignment_v2_components(info) if self.config.reward_version == 'alignment_v2'
                      else self._alignment_components(info))
        return float(sum(components[key] for key in ('bore_alignment', 'axial', 'tilt', 'carrier')))

    def _observation(self, info):
        proprio = []
        controller = []
        for side, arm in self.sim.arms.items():
            proprio.extend(np.r_[arm.q, self.data.qpos[arm.finger_qadr[0]],
                                 self.data.qvel[arm.arm_dofadr], self.data.qvel[arm.finger_dofadr[0]]])
            p, r = self.targets[side]
            controller.extend(np.r_[self.commands[side], self.bias[side], self.velocity[side],
                                     p, r.ravel(), self.grips[side]])
        controller.extend([self.steps / self.config.max_steps,
                           self.seated_steps / self.config.success_steps])
        hand_p, hand_r = self._in_hand()
        gear_r = self.data.xmat[self.gear_id].reshape(3, 3)
        parts = self.sim.manifest['parts']
        pin_in_gear = (np.asarray(parts[self.pin_name]['assembled_origin'])
                       - parts['gear_3']['assembled_origin'])
        gear_target = self.data.xpos[self.gear_id] + gear_r @ pin_in_gear
        carrier_target, _ = pin_target_pose(self.sim, self.pin_name)
        task = np.r_[info['pin_delta_m'], self.data.xmat[self.pin_id].reshape(3, 3)[:, 2],
                     self.data.xpos[self.carrier_id] - self.carrier_reference,
                     self.data.xmat[self.carrier_id].reshape(3, 3)[:, 2], hand_p, hand_r.ravel(),
                     info['axial_error_mm'] / 1000, np.deg2rad(info['tilt_deg']), self.peak_load,
                     gear_target - carrier_target, gear_r[:, 2]]
        return dict(proprioception=np.array(proprio, dtype=np.float32),
                    task=task.astype(np.float32), controller=np.array(controller, dtype=np.float32))

    def step(self, action):
        if not self._ready:
            raise RuntimeError('Call reset before stepping, including after episode end')
        action = np.asarray(action, dtype=float)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError('Action must contain 14 finite values')
        action = np.clip(action, -1, 1)
        ik_errors = {}
        for i, (side, arm) in enumerate(self.sim.arms.items()):
            command = action[i * 7:(i + 1) * 7]
            p, r = self.targets[side]
            p0, r0 = self.origins[side]
            step_m, step_rad = (.001, np.deg2rad(.5)) if side == 'left' else (.00025, np.deg2rad(.1))
            extent = np.array([.04, .04, .065]) if side == 'left' else np.full(3, self.config.holder_workspace_m)
            p = np.clip(p + command[:3] * step_m, p0 - extent, p0 + extent)
            r_new = rotation(command[3:6] * step_rad) @ r
            angle = np.arccos(np.clip((np.trace(r_new @ r0.T) - 1) / 2, -1, 1))
            if angle <= np.deg2rad(15 if side == 'left' else 3):
                r = r_new
            self.targets[side] = p, r
            self.grips[side] = float(np.clip(self.grips[side] + command[6] * .05, 0, 1))
            desired, pe, re = arm.ik(p, mat_to_quat(r), q_init=self.commands[side],
                                       iters=40, max_joint_step=.03, posture_gain=0,
                                       min_damping=.001, pos_tol=.00002)
            ik_errors[side] = [float(pe), float(re)]
            self.bias[side] = np.clip(self.bias[side] + .1 * (desired - arm.q), -.012, .012)
            speed = np.clip((desired + self.bias[side] - self.commands[side]) / .05, -.6, .6)
            speed = np.clip(speed, self.velocity[side] - .1, self.velocity[side] + .1)
            self.commands[side] = np.clip(self.commands[side] + .05 * speed, *arm.arm_range.T)
            self.velocity[side] = speed
            arm.set_arm_ctrl(self.commands[side])
            arm.set_gripper(self.grips[side])
        self.peak_load = 0.
        # Sample loads through the control interval so brief impacts count.
        pending = {s: self.data.ctrl[a.act_ids].copy() for s, a in self.sim.arms.items()}
        previous = {s: a.applied.copy() for s, a in self.sim.arms.items()}
        for tick in range(1, 51):
            for side, arm in self.sim.arms.items():
                self.data.ctrl[arm.act_ids] = previous[side] + (pending[side] - previous[side]) * tick / 50
            mujoco.mj_step(self.model, self.data)
            self.peak_load = max(self.peak_load, self._contact_load())
        self.sim.sync_control_state()
        self.steps += 1
        info = self._info()
        self.seated_steps = self.seated_steps + 1 if info['is_success'] else 0
        success = self.seated_steps >= self.config.success_steps
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        pin_below_table = self.data.xpos[self.pin_id, 2] < TABLE_Z - .005
        escaped = np.linalg.norm(info['pin_delta_m'][:2]) > .05
        failure = ('nonfinite_state' if not finite else 'excess_contact_force'
                   if self.peak_load > self.config.contact_limit_n else 'carrier_displaced'
                   if not info['carrier_supported'] else 'pin_dropped'
                   if pin_below_table or escaped else None)
        terminated = bool(success or failure)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)
        score = self._reward_score(info)
        potential_delta = score - self._potential
        reward = potential_delta - .002 - .001 * float(action @ action)
        reward -= .001 * self.peak_load
        reward += 10. if success and not failure else -5. if failure else 0.
        self._potential = score
        info.update(is_success=bool(success and not failure), failure=failure,
                    ik_errors=ik_errors, seated_steps=self.seated_steps)
        if self.config.reward_version in ('alignment_v1', 'alignment_v2'):
            info['reward_version'] = self.config.reward_version
            info['reward_components'] = dict(
                potential_delta=float(potential_delta),
                time=-.002, action=-.001 * float(action @ action),
                contact=-.001 * self.peak_load,
                terminal=10. if success and not failure else -5. if failure else 0.,
                potential=(self._alignment_v2_components(info) if self.config.reward_version == 'alignment_v2'
                           else self._alignment_components(info)))
        self._ready = not (terminated or truncated)
        if self.render_mode == 'human':
            self.render()
        return self._observation(info), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            if self._viewer is None:
                from mujoco import viewer as mj_viewer
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
                self._viewer.opt.geomgroup[3] = 0
            self._viewer.sync()
        elif self.render_mode == 'rgb_array':
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            option = mujoco.MjvOption()
            option.geomgroup[3] = 0
            self._renderer.update_scene(self.data, camera='overhead', scene_option=option)
            return self._renderer.render().copy()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register('Panthera-BimanualPinInsertion-v0', entry_point=BimanualPinInsertionEnv)
