"""Physical MuJoCo pin manipulation behind the collection backend interface."""
from __future__ import annotations

import json
from collection.pin_regrasp import rotation_exp

import mujoco
import numpy as np

from sim.actuator_env import ASSETS, TABLE_Z
from sim.panthera_env import GRIPPER_OPEN, mat_to_quat
from sim.bimanual_actuator_env import BimanualActuatorSim


class MujocoPinBackend:
    """Manipulate a table-supported pin while the second arm holds the carrier.

    Opening is per-finger travel in metres. Only reset writes joint/object
    poses; commands run the original position servos and contact dynamics.
    """

    open_width = .025
    closed_width = 0.
    lift_distance = .03

    def __init__(self, hz=30, image_size=128, images=True):
        if not np.isfinite(hz) or not 1 <= hz <= 1000 or image_size < 1:
            raise ValueError('hz and image_size must be positive')
        self.sim = BimanualActuatorSim(layout='side_by_side')
        self.arm = self.sim.arms['left']
        self.right = self.sim.arms['right']
        self.profile = json.loads((ASSETS / 'pin_regrasp_profile.json').read_text())
        self._steps = max(1, round(1 / (hz * self.sim.model.opt.timestep)))
        self.hz = 1 / (self._steps * self.sim.model.opt.timestep)
        self.image_size, self.images = int(image_size), images
        self._renderer = None
        self._pin = self.sim.model.body('pin_3').id
        self.sim._cube_geoms = {g: 'pin_3' for g in range(self.sim.model.ngeom)
                                if self.sim.model.geom_bodyid[g] == self._pin}
        self._pin_index = self.sim.object_names.index('pin_3')
        self._qctrl = self.arm.q.copy()
        self._bias = np.zeros(6)
        self._opening = self.open_width
        self._ready = False

        self._robot_bodies = set()
        for body in range(self.sim.model.nbody):
            ancestor = body
            while ancestor:
                if self.sim.model.body(ancestor).name in ('link1', 'right_link1'):
                    self._robot_bodies.add(body)
                    break
                ancestor = self.sim.model.body_parentid[ancestor]
        self._left_pads = {self.sim.model.geom(n).id for n in ('L_finger_pad', 'R_finger_pad')}
        self._right_pads = {self.sim.model.geom(n).id for n in ('right_L_finger_pad', 'right_R_finger_pad')}
        self._carrier = self.sim.model.body('large_carrier').id

    def reset(self, seed: int):
        s = self.sim
        rng = np.random.default_rng(seed)
        self._ready = False
        s.reset(randomize=False)
        sample = self.profile['samples'][seed % len(self.profile['samples'])]
        base = s.data.xpos[s.model.body('left_base').id]
        xy = (base + np.asarray(sample['left']['position_base_m']))[:2] + rng.uniform(-.02, .02, 2)
        yaw = rng.uniform(-np.pi, np.pi)
        s.set_part_pose('pin_3', [*xy, TABLE_Z + .0002], [np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        # Keep unused loose parts clear of the two active grasps.
        for i, name in enumerate(('pin_1', 'pin_2', 'pin_4', 'gear_1', 'gear_2', 'gear_4')):
            s.set_part_pose(name, [.54, -.28 + .065*i, TABLE_Z + .0002])
        fixture = s.model.body('small_carrier').id
        s.model.body_pos[fixture, :2] = [.28, -.12]
        s.model.geom_pos[s.model.geom('fixture').id, :2] = [.28, -.12]
        s.model.body_quat[fixture] = mat_to_quat(rotation_exp(np.array([0., 0., -np.pi/2])))
        mujoco.mj_forward(s.model, s.data)
        for name in ('large_carrier', 'gear_3'):
            s.set_part_pose(name, *s.assembled_pose(name))
        holder = sample
        holder_r = np.asarray(holder['right']['rotation_base'])
        horizontal = holder_r[:, 0].copy(); horizontal[2] = 0
        horizontal /= np.linalg.norm(horizontal)
        pitch = np.arcsin(-holder_r[2, 0])
        carrier = s.model.body('large_carrier').id
        mujoco.mj_forward(s.model, s.data)
        holder_p = s.data.xpos[carrier] + horizontal*.010*np.sin(pitch) + [0, 0, .004]
        rotation_jitter = rng.uniform(-3, 3, 3)
        rotation = np.asarray(sample['left']['rotation_base']) @ rotation_exp(np.deg2rad(rotation_jitter))
        target = np.r_[xy, TABLE_Z + .0132]
        for arm, position, orient, width, recorded in (
                (self.right, holder_p, holder_r, .027, holder['right']),
                (self.arm, target, rotation, self.open_width, sample['left'])):
            q, pe, re = arm.ik(position, mat_to_quat(orient), q_init=np.asarray(recorded['joints_rad']),
                               max_joint_step=None, iters=300, posture_gain=0, min_damping=.001)
            if pe > .0002 or re > .005:
                raise RuntimeError(f'Reset grasp IK failed: {pe:g} m, {re:g} rad')
            s.data.qpos[arm.arm_qadr] = q
            s.data.qpos[arm.finger_qadr] = [width, -width]
            arm.set_arm_ctrl(q, immediate=True)
            arm.set_gripper(0)
        s.data.qvel[:] = 0
        self._qctrl, self._bias = self.arm.q.copy(), np.zeros(6)
        self._holder_q = self.right.q.copy()
        self._ready = False
        mujoco.mj_forward(s.model, s.data)
        for i in range(round(self.hz * 1.2)):
            self.command(target, rotation, self.open_width * max(0., 1-i/(self.hz*.6)))
        supported = any(
            self.sim.model.geom(c.geom1).name == 'table_top' or
            self.sim.model.geom(c.geom2).name == 'table_top'
            for c in s.data.contact
            if self._pin in (s.model.geom_bodyid[c.geom1], s.model.geom_bodyid[c.geom2]))
        if not supported:
            raise RuntimeError('Reset pin is not supported by the table')
        if not self.grasped():
            raise RuntimeError('Reset did not establish loaded bilateral pin contact')
        self._reference_z = float(s.data.xpos[self._pin, 2])
        self._ready = True
        if not self._holder_grasped():
            raise RuntimeError("Right arm failed to grasp carrier")
        obs = self.observe()
        return {'seed': seed, 'anchor_position': obs['ee_position'].tolist(),
                'anchor_rotation': obs['ee_rotation'].tolist(), 'pin_yaw': float(yaw),
                'rotation_jitter_deg': rotation_jitter.tolist(), 'table_height': TABLE_Z, 'control_hz': self.hz,
                'pin_xy': xy.tolist(), 'xy_jitter_m': .02, 'approach_retreat_direction': (-np.asarray(sample['approach_direction_base'])).tolist(),
                'profile_episode': sample['episode_id'], 'profile_time_s': sample['time_s'],
                'layout': 'side_by_side', 'right_task': 'hold_large_carrier',
                'right_profile_rotation': holder_r.tolist(), 'right_target_position': holder_p.tolist(),
                'carrier_grasp_retargeting': 'Recorded right orientation at table-supported assembled carrier'}

    def command(self, position, rotation, opening):
        s = self.arm
        position, rotation = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
        if position.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError('Expected finite position (3,) and rotation (3,3)')
        if not 0 <= opening <= self.open_width:
            raise ValueError('Opening is per-finger travel in [0, open_width] metres')
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < .999:
            raise ValueError('Rotation must be a proper orthonormal matrix')
        quat = mat_to_quat(rotation)
        q, pe, re = s.ik(position, quat, q_init=self._qctrl, iters=100,
                         max_joint_step=.09, posture_gain=0, pos_tol=1e-6,
                         rot_tol=1e-5, min_damping=.001)
        if pe > .001 or re > .025:
            raise RuntimeError(f'Command IK failed: {pe:g} m, {re:g} rad')
        self._bias = np.clip(self._bias + .25 * (self._qctrl-s.q), -.015, .015)
        self._qctrl, self._opening = q, float(opening)
        s.set_arm_ctrl(q + self._bias)
        s.set_gripper(opening / GRIPPER_OPEN)
        s.step(self._steps)
        error = s._ee_error(position, quat)
        if not np.isfinite(s.data.qpos).all() or np.linalg.norm(error[:3]) > .015 or np.linalg.norm(error[3:]) > .15:
            raise RuntimeError(f'Physical end-effector tracking failed: {np.linalg.norm(error[:3]):g} m, {np.linalg.norm(error[3:]):g} rad')
        if self._ready and not self._holder_grasped():
            raise RuntimeError('Right arm lost the carrier grasp')
        for c in s.data.contact:
            b1, b2 = s.model.geom_bodyid[[c.geom1, c.geom2]]
            if c.dist < -.001 and (b1 in self._robot_bodies or b2 in self._robot_bodies):
                intended = ((self._pin in (b1, b2) and
                             (c.geom1 in self._left_pads or c.geom2 in self._left_pads)) or
                            (self._carrier in (b1, b2) and
                             (c.geom1 in self._right_pads or c.geom2 in self._right_pads)))
                if not intended:
                    raise RuntimeError('Unexpected robot collision')

    def observe(self):
        s = self.sim
        result = {'time': np.asarray(s.data.time),
                  'state': np.concatenate([np.r_[a.q, a.dq, s.data.qpos[a.finger_qadr]] for a in s.arms.values()]),
                  'right_grasped': np.asarray(self._holder_grasped()),
                  'carrier_pose': np.r_[s.data.xpos[s.model.body('large_carrier').id], s.data.xquat[s.model.body('large_carrier').id]],
                  'right_ee_position': self.right.ee_pos(),
                  'right_ee_rotation': s.data.site_xmat[self.right.ee_site].reshape(3, 3).copy(),
                  'ee_position': self.arm.ee_pos(),
                  'ee_rotation': s.data.site_xmat[self.arm.ee_site].reshape(3, 3).copy(),
                  'object_pose': np.r_[s.data.xpos[self._pin], s.data.xquat[self._pin]],
                  'grasped': np.asarray(self.grasped())}
        if self.images:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(s.model, self.image_size, self.image_size)
            option = mujoco.MjvOption()
            option.geomgroup[3] = 0
            option.sitegroup[:] = 0
            for camera in s.cameras:
                self._renderer.update_scene(s.data, camera=camera, scene_option=option)
                result['image_' + camera] = self._renderer.render().copy()
            overview = mujoco.MjvCamera()
            overview.lookat[:] = [.22, 0., .24]
            overview.distance, overview.azimuth, overview.elevation = 1.25, 135., -32.
            self._renderer.update_scene(s.data, camera=overview, scene_option=option)
            result['image_overview'] = self._renderer.render().copy()
        return result

    def grasped(self):
        return bool(np.all(self.sim.pad_normal_forces()[self._pin_index] > .01))

    def _holder_grasped(self):
        s = self.sim
        carrier = s.model.body("large_carrier").id
        forces = np.zeros(2)
        pads = [s.model.geom(n).id for n in ("right_L_finger_pad", "right_R_finger_pad")]
        wrench = np.zeros(6)
        for i, c in enumerate(s.data.contact):
            for object_geom, pad in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if s.model.geom_bodyid[object_geom] == carrier and pad in pads and c.efc_address >= 0:
                    mujoco.mj_contactForce(s.model, s.data, i, wrench)
                    forces[pads.index(pad)] += max(0., wrench[0])
        return bool(np.all(forces > .01))

    def verify_lift(self, min_height):
        return self.grasped() and float(self.sim.data.xpos[self._pin, 2])-self._reference_z >= min_height

    def supported(self):
        """Whether the pin has an active contact with the table surface."""
        s = self.sim
        return any(c.efc_address >= 0 and
                   self._pin in (s.model.geom_bodyid[c.geom1], s.model.geom_bodyid[c.geom2]) and
                   'table_top' in (s.model.geom(c.geom1).name, s.model.geom(c.geom2).name)
                   for c in s.data.contact)

    def stop(self):
        self.arm.set_arm_ctrl(self.arm.q)
        self.arm.set_gripper(self._opening / GRIPPER_OPEN)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
