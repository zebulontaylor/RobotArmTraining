"""Two physical Panthera arms and three cameras for place-then-push assembly.

The synthetic wrist names are explicit arm names: the real USB camera-to-arm
assignment is uncalibrated. This scene is deliberately separate from legacy
single-arm recordings and checkpoints.
"""
import copy
import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.actuator_env import ActuatorSim, ASSETS, TABLE_Z
from sim.panthera_env import PantheraSim, ARM_JOINTS
from sim.real_setup import DEFAULT_SETUP, apply_photo_setup

CAMERAS = ('overhead', 'left_wrist', 'right_wrist')
STATE_NAMES = tuple(f'{side}.{name}' for side in ('left', 'right')
                    for name in (*ARM_JOINTS, 'gripper_open_m'))


def build_bimanual_scene(layout='opposed', setup_profile=None):
    if layout not in ('opposed', 'side_by_side'):
        raise ValueError(f'Unknown bimanual layout: {layout}')
    setup_profile = setup_profile or (DEFAULT_SETUP if layout == 'side_by_side' else 'legacy')
    if setup_profile not in ('legacy', DEFAULT_SETUP):
        raise ValueError(f'Unknown setup profile: {setup_profile}')
    if setup_profile != 'legacy' and layout != 'side_by_side':
        raise ValueError('The photo setup requires the side_by_side layout')
    # Read the checked-in single-arm asset; never rewrite legacy scene files.
    root = ET.parse(ASSETS/'scene_pins.xml').getroot()
    root.set('model', 'panthera_bimanual_place_push')
    world = root.find('worldbody')
    for geom in list(world.findall('geom')):
        if geom.get('name', '').endswith('_rack'):
            world.remove(geom)
    for body in world.findall('body'):
        if body.get('name', '').startswith('gear_'):
            position = body.get('pos').split(); position[2] = str(TABLE_Z+.0002)
            body.set('pos', ' '.join(position))
    robot = world.find("body[@name='link1']")
    # Training cameras should see solid arms, including their occlusions.
    for geom in world.findall(".//geom[@class='visual']"):
        if geom.get('rgba'):
            rgba = geom.get('rgba').split(); rgba[-1] = '1'
            geom.set('rgba', ' '.join(rgba))
    second = copy.deepcopy(robot)
    names = {e.get('name'): 'right_'+e.get('name') for e in robot.iter() if e.get('name')}
    for e in second.iter():
        for key, value in list(e.attrib.items()):
            if key == 'name':
                e.set(key, names[value])
    # Opposed bases sharing a table, in a synthetic, uncalibrated frame.
    base = ET.SubElement(world, 'body', name='right_base', pos='.72 .16 0', quat='0 0 0 1')
    for geom in world.findall('geom'):
        if geom.get('class') in ('visual', 'collision'):
            base.append(copy.deepcopy(geom))
    base.append(second)
    second.find(".//camera[@name='right_wrist']").set('name', 'right_wrist')
    robot.find(".//camera[@name='wrist']").set('name', 'left_wrist')
    for camera in list(world.findall('camera')):
        if camera.get('name') != 'overhead':
            world.remove(camera)
    overhead = world.find("camera[@name='overhead']")
    overhead.set('pos', '.40 -.40 .65')
    overhead.set('xyaxes', '1 0 0 0 .8 .6')
    if layout == 'side_by_side':
        root.set('model', 'panthera_side_by_side_pin_insertion')
        # Both bases stand on the same table edge, facing the shared workspace.
        base.set('pos', f'0 -.29 {TABLE_Z}')
        base.set('quat', '1 0 0 0')
        index = list(world).index(robot)
        world.remove(robot)
        left = ET.Element('body', name='left_base', pos=f'0 .29 {TABLE_Z}')
        for geom in list(world.findall('geom')):
            if geom.get('class') in ('visual', 'collision'):
                world.remove(geom)
                left.append(geom)
        left.append(robot)
        world.insert(index, left)
        table = world.find("geom[@name='table_top']")
        table.set('pos', '.23 0 .025')
        table.set('size', '.45 .55 .025')
        tower = ET.SubElement(world, 'body', name='center_tower', pos=f'-.04 0 {TABLE_Z}')
        for name, pos, size in (
                ('tower_foot', '0 0 .01', '.075 .075 .01'),
                ('tower_column', '0 0 .36', '.025 .025 .35'),
                ('tower_head', '.015 0 .73', '.045 .055 .035')):
            ET.SubElement(tower, 'geom', name=name, type='box', pos=pos, size=size,
                          rgba='.16 .19 .22 1', contype='1', conaffinity='3')
        ET.SubElement(tower, 'geom', name='tower_lens', type='cylinder',
                      pos='.064 0 .73', size='.012 .006', quat='.707107 0 .707107 0',
                      rgba='.04 .07 .10 1', contype='0', conaffinity='0')
        camera_position = np.array([1.20, .10, .85])
        backward = camera_position - [.16, 0, .38]
        backward /= np.linalg.norm(backward)
        right = np.cross([0, 0, 1.], backward)
        right /= np.linalg.norm(right)
        up = np.cross(backward, right)
        overhead.set('pos', ' '.join(map(str, camera_position)))
        overhead.set('xyaxes', ' '.join(map(str, np.r_[right, up])))
        overhead.set('fovy', '55')
        root.find('statistic').set('center', '.18 0 .35')
        root.find('statistic').set('extent', '.85')
    # Exactly one overhead camera plus a camera attached to each wrist.
    for section in ('actuator', 'equality', 'contact'):
        parent = root.find(section)
        for e in list(parent):
            clone = copy.deepcopy(e)
            for key, value in list(clone.attrib.items()):
                if key in ('name', 'joint', 'joint1', 'joint2', 'body1', 'body2'):
                    clone.set(key, 'right_base' if value == 'world' else 'right_'+value)
            parent.append(clone)
    if layout == 'side_by_side':
        root.find("contact/exclude[@body1='world']").set('body1', 'left_base')
    # Keyframe tails would otherwise initialize the second arm at zero.
    root.remove(root.find('keyframe'))
    home = [0., 1.2, 1.4, 0., 0., 0., .04, -.04]
    free_poses = []
    for body in world.findall('body'):
        if body.find('freejoint') is not None:
            free_poses.extend(map(float, body.get('pos').split()))
            free_poses.extend(map(float, body.get('quat', '1 0 0 0').split()))
    ET.SubElement(ET.SubElement(root, 'keyframe'), 'key', name='home',
                  qpos=' '.join(map(str, home+free_poses+home)),
                  ctrl=' '.join(map(str, home[:7]+home[:7])))
    if setup_profile == DEFAULT_SETUP:
        apply_photo_setup(root)
    ET.indent(root)
    path = ASSETS / ('scene_bimanual.xml' if layout == 'opposed'
                     else ('scene_bimanual_side_by_side.xml' if setup_profile == 'legacy'
                           else f'scene_bimanual_{setup_profile}.xml'))
    # Independent seed runs may construct the same immutable scene at once.
    with tempfile.NamedTemporaryFile(mode='w', dir=ASSETS, suffix='.xml', delete=False) as f:
        temporary = f.name
        ET.ElementTree(root).write(f, encoding='unicode')
    os.replace(temporary, path)
    return path


class Arm(PantheraSim):
    """An IK/control view into one arm of a shared model and physics state."""
    def __init__(self, sim, side):
        self.sim, self.side = sim, side
        self.model, self.data = sim.model, sim.data
        prefix = '' if side == 'left' else 'right_'
        joints = [self.model.joint(prefix+n).id for n in ARM_JOINTS]
        self.arm_qadr = self.model.jnt_qposadr[joints].copy()
        self.arm_dofadr = self.model.jnt_dofadr[joints].copy()
        self.arm_range = self.model.jnt_range[joints].copy()
        fingers = [self.model.joint(prefix+n+'_finger_joint').id for n in ('L', 'R')]
        self.finger_qadr = self.model.jnt_qposadr[fingers].copy()
        self.finger_dofadr = self.model.jnt_dofadr[fingers].copy()
        self.act_ids = np.array([self.model.actuator(prefix+n).id for n in ARM_JOINTS])
        self.grip_act = self.model.actuator(prefix+'gripper').id
        self.ee_site = self.model.site(prefix+'grip_site').id
        self._q_nominal = np.array([0., 1.2, 1.4, 0., 0., 0.])
        self._ik_data = mujoco.MjData(self.model)
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros_like(self._jacp)
        self.manifest = sim.manifest
        self.applied = self.data.ctrl[self.act_ids].copy()

    def set_arm_ctrl(self, q, *, immediate=False):
        self.data.ctrl[self.act_ids] = np.clip(q, *self.arm_range.T)
        if immediate:
            self.applied = self.data.ctrl[self.act_ids].copy()

    def step(self, n=1):
        self.sim.step(n)


class BimanualActuatorSim(ActuatorSim):
    cameras = CAMERAS
    state_names = STATE_NAMES

    def __init__(self, seed=0, xy_jitter=.002, yaw_jitter_deg=2., *, layout='opposed', setup_profile=None):
        self.layout = layout
        self.setup_profile = setup_profile or (DEFAULT_SETUP if layout == 'side_by_side' else 'legacy')
        super().__init__(seed, xy_jitter, yaw_jitter_deg, with_pins=True,
                         scene=build_bimanual_scene(layout, self.setup_profile))
        self.arms = {side: Arm(self, side) for side in ('left', 'right')}
        self.stations = dict(self.stations, large_carrier=(.43, .12), gear_3=(.28, -.12))
        self.reset()

    def reset(self, *, randomize=True, rng=None):
        super().reset(randomize=randomize, rng=rng)
        # Parent initialization calls reset before arm views exist.
        for side in ('left', 'right'):
            prefix = '' if side == 'left' else 'right_'
            for name, value in zip(ARM_JOINTS, [0., 1.2, 1.4, 0., 0., 0.]):
                j = self.model.joint(prefix+name).id
                self.data.qpos[self.model.jnt_qposadr[j]] = value
                self.data.ctrl[self.model.actuator(prefix+name).id] = value
            for name, value in (('L', .04), ('R', -.04)):
                j = self.model.joint(prefix+name+'_finger_joint').id
                self.data.qpos[self.model.jnt_qposadr[j]] = value
            self.data.ctrl[self.model.actuator(prefix+'gripper').id] = .04
        self.sync_control_state()
        mujoco.mj_forward(self.model, self.data)

    def sync_control_state(self):
        super().sync_control_state()
        for arm in getattr(self, 'arms', {}).values():
            arm.applied = self.data.ctrl[arm.act_ids].copy()

    def state(self):
        return np.concatenate([np.r_[a.q, self.data.qpos[a.finger_qadr[0]]]
                               for a in self.arms.values()])

    def action(self):
        return np.concatenate([np.r_[self.data.ctrl[a.act_ids], self.data.ctrl[a.grip_act]]
                               for a in self.arms.values()])

    def step(self, n=1):
        if n < 0 or int(n) != n:
            raise ValueError('Physics step count must be a nonnegative integer')
        if not n:
            return
        targets = {s: self.data.ctrl[a.act_ids].copy() for s, a in self.arms.items()}
        for tick in range(1, int(n)+1):
            for s, a in self.arms.items():
                self.data.ctrl[a.act_ids] = a.applied+(targets[s]-a.applied)*(tick/n)
            mujoco.mj_step(self.model, self.data)
        for s, a in self.arms.items():
            self.data.ctrl[a.act_ids] = targets[s]
            a.applied = targets[s]
        self._applied_arm_ctrl = self.data.ctrl[:6].copy()
