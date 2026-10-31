"""Approximate scene appearance and camera mounts from the hardware photographs.

The profile describes a visual reference. Its camera poses and fields of view
are estimates, so they do not supply real-image triangulation calibration.
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_SETUP = 'photo_20261027'
PROFILE_PATH = Path(__file__).resolve().parents[1] / 'config/real_setup_20261027.json'


def load_setup():
    """Read the portable profile and its measurement provenance."""
    return json.loads(PROFILE_PATH.read_text())


def _numbers(values):
    return ' '.join(f'{float(value):.9g}' for value in values)


def _axes(pitch_deg):
    pitch = np.deg2rad(pitch_deg)
    return np.array([0., -1., 0.]), np.array([np.sin(pitch), 0., np.cos(pitch)])


def _visual(parent, name, kind, position, size, rgba, **extra):
    existing = parent.find(f"geom[@name='{name}']")
    if existing is None:
        existing = ET.SubElement(parent, 'geom', name=name)
    existing.attrib.update(type=kind, pos=_numbers(position), size=_numbers(size),
                           rgba=_numbers(rgba), group='2', mass='0',
                           contype='0', conaffinity='0', **extra)
    return existing


def _camera_mount(parent, prefix, eye, pitch, board_size):
    right, up = _axes(pitch)
    backward = np.cross(right, up)
    axes = _numbers(np.r_[right, up])
    eye = np.asarray(eye)
    black = [.035, .038, .043, 1.]
    _visual(parent, f'{prefix}_board', 'box', eye + backward * .014,
            [board_size, board_size, .002], [.18, .20, .19, 1.], xyaxes=axes)
    _visual(parent, f'{prefix}_lens', 'cylinder', eye + backward * .006,
            [.009, .005], black, xyaxes=axes)
    _visual(parent, f'{prefix}_bracket', 'box', eye + backward * .025,
            [board_size + .004, board_size + .004, .007], black, xyaxes=axes)


def bench_texture(rng, size=256):
    """Generate honey-colored staggered butcher-block strips with fine grain."""
    yy, xx = np.mgrid[:size, :size]
    texture = np.empty((size, size, 3), dtype=float)
    strip_height = max(8, size // 10)
    for row in range((size + strip_height - 1) // strip_height):
        strip = yy // strip_height == row
        length = max(20, size // 3)
        offset = int(rng.integers(0, length))
        sections = (xx + offset) // length
        for section in range(int(sections.max()) + 1):
            mask = strip & (sections == section)
            base = np.array([188., 139., 78.]) + rng.normal(0., [12., 10., 7.])
            grain = 3.5 * np.sin(yy * 2.7 + np.sin(xx / 35.) * 1.7 + row)
            grain += 1.8 * np.sin(yy * 8.1 + xx / 54.)
            noise = rng.normal(0., 1.1, (size, size))
            texture[mask] = (base + (grain + noise)[..., None])[mask]
        seam = strip & (((xx + offset) % length == 0) | (yy % strip_height == 0))
        texture[seam] *= .83
    return np.clip(texture, 0, 255).astype(np.uint8)


def _bench_material(root):
    asset = root.find('asset')
    if asset.find("texture[@name='photo_bench']") is None:
        ET.SubElement(asset, 'texture', name='photo_bench', type='2d',
                      file='textures/bench_photo_20261027.png')
    if asset.find("material[@name='photo_bench']") is None:
        ET.SubElement(asset, 'material', name='photo_bench', texture='photo_bench',
                      texrepeat='2 2', texuniform='true', rgba='1 1 1 1',
                      specular='.12', shininess='.2')
    table = root.find("worldbody/geom[@name='table_top']")
    table.set('material', 'photo_bench')
    table.set('rgba', '1 1 1 1')
    if asset.find("material[@name='photo_metal']") is None:
        ET.SubElement(asset, 'material', name='photo_metal', rgba='.55 .57 .59 1',
                      specular='.55', shininess='.4')
    for geom in root.findall('worldbody/.//geom'):
        if geom.get('name', '').endswith('_visual') and geom.get('mesh', '').startswith(
                ('pin_', 'gear_', 'large_carrier', 'small_carrier')):
            geom.set('rgba', '.55 .57 .59 1')
            geom.set('material', 'photo_metal')


def _robot_colors(world):
    for geom in world.findall('.//geom'):
        mesh = geom.get('mesh', '')
        if geom.get('class') != 'visual':
            continue
        if mesh == 'gripper_center':
            geom.set('rgba', '.055 .058 .063 1')
        elif mesh in ('L_finger', 'R_finger'):
            geom.set('rgba', '.83 .85 .81 1')
        elif mesh in ('base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6'):
            geom.set('rgba', '.65 .68 .70 1')
    for body in world.findall('.//body'):
        for pad in list(body.findall('geom')):
            if pad.get('name', '').endswith('_finger_pad'):
                _visual(body, pad.get('name') + '_visual', 'box',
                        np.fromstring(pad.get('pos'), sep=' '),
                        np.fromstring(pad.get('size'), sep=' '), [.035, .038, .043, 1.])


def apply_photo_setup(root):
    """Update a side-by-side scene XML after its robot and keyframe exist.

    Added mounts and controller boxes have zero mass and no contacts. Existing
    static tower colliders follow the estimated tower envelope. Joint frames,
    arm inertials, object collisions and keyframe values are left in place.
    """
    profile = load_setup()
    world = root.find('worldbody')
    _bench_material(root)
    _robot_colors(world)
    headlight = root.find('visual/headlight')
    headlight.set('ambient', '.35 .35 .35')
    headlight.set('diffuse', '.45 .45 .45')
    headlight.set('specular', '.15 .15 .15')
    for light in world.findall('light'):
        light.set('diffuse', '.45 .45 .45')
        light.set('specular', '.25 .25 .25')
    tower = world.find("body[@name='center_tower']")
    if tower is None:
        raise ValueError('The photo setup requires the side-by-side central tower')
    t = profile['tower']
    for name, position, size in (
        ('tower_foot', [0., 0., .008], [.075, .075, .008]),
        ('tower_column', [0., 0., t['column_top_m'] / 2],
         [t['depth_m'] / 2, t['width_m'] / 2, t['column_top_m'] / 2]),
    ):
        geom = tower.find(f"geom[@name='{name}']")
        geom.set('pos', _numbers(position))
        geom.set('size', _numbers(size))
        geom.set('rgba', '.035 .038 .043 1')
    eye_local = np.asarray(t['optical_center_local_m'])
    right, up = _axes(t['pitch_down_deg'])
    backward = np.cross(right, up)
    head = tower.find("geom[@name='tower_head']")
    head.set('pos', _numbers(eye_local + backward * .025))
    head.set('size', '.025 .025 .007')
    head.set('xyaxes', _numbers(np.r_[right, up]))
    head.attrib.pop('quat', None)
    head.set('rgba', '.035 .038 .043 1')
    lens = tower.find("geom[@name='tower_lens']")
    lens.set('pos', _numbers(eye_local + backward * .006))
    lens.set('size', '.009 .005')
    lens.set('xyaxes', _numbers(np.r_[right, up]))
    lens.attrib.pop('quat', None)
    lens.set('rgba', '.025 .028 .033 1')
    _visual(tower, 'tower_camera_board', 'box', eye_local + backward * .014,
            [.021, .021, .002], [.18, .20, .19, 1.], xyaxes=_numbers(np.r_[right, up]))
    overhead = world.find("camera[@name='overhead']")
    tower_pos = np.fromstring(tower.get('pos'), sep=' ')
    overhead.set('pos', _numbers(tower_pos + eye_local))
    overhead.set('xyaxes', _numbers(np.r_[right, up]))
    overhead.attrib.pop('quat', None)
    overhead.set('fovy', str(t['fovy_deg']))
    w = profile['wrist_cameras']
    for body_name, camera_name in (('link6', 'left_wrist'), ('right_link6', 'right_wrist')):
        body = world.find(f".//body[@name='{body_name}']")
        camera = body.find(f"camera[@name='{camera_name}']")
        right, up = _axes(w['pitch_down_deg'])
        camera.set('pos', _numbers(w['optical_center_link6_m']))
        camera.set('xyaxes', _numbers(np.r_[right, up]))
        camera.attrib.pop('quat', None)
        camera.set('fovy', str(w['fovy_deg']))
        _camera_mount(body, camera_name + '_camera', w['optical_center_link6_m'],
                      w['pitch_down_deg'], .018)
        _visual(body, camera_name + '_support', 'box', [.015, 0., .047],
                [.013, .022, .045], [.035, .038, .043, 1.])
    for side, y in (('left', .17), ('right', -.17)):
        _visual(world, f'{side}_controller_box', 'box', [-.025, y, tower_pos[2] + .022],
                [.067, .045, .022], [.035, .038, .043, 1.])
    return profile


def configure_photo_cameras(model):
    """Set nominal photo-profile optics after a perception model is compiled."""
    import mujoco

    profile = load_setup()
    tower_id = model.body('center_tower').id
    t = profile['tower']
    eye = model.body_pos[tower_id] + np.asarray(t['optical_center_local_m'])
    for name, position, pitch, fovy in (
        ('overhead', eye, t['pitch_down_deg'], t['fovy_deg']),
        *[(name, profile['wrist_cameras']['optical_center_link6_m'],
           profile['wrist_cameras']['pitch_down_deg'], profile['wrist_cameras']['fovy_deg'])
          for name in ('left_wrist', 'right_wrist')],
    ):
        right, up = _axes(pitch)
        rotation = np.column_stack([right, up, np.cross(right, up)])
        camera = model.camera(name).id
        model.cam_pos[camera] = position
        mujoco.mju_mat2Quat(model.cam_quat[camera], rotation.ravel())
        model.cam_fovy[camera] = fovy
    right, up = _axes(t['pitch_down_deg'])
    return dict(profile=DEFAULT_SETUP, source_profile='config/real_setup_20261027.json',
                tower_height_m=t['optical_center_local_m'][2],
                tabletop_z_m=float(model.body_pos[tower_id, 2]),
                overhead_eye_world_m=eye.tolist(),
                overhead_lookat_world_m=(eye - np.cross(right, up) * .4).tolist(),
                overhead_fovy_deg=t['fovy_deg'],
                calibration='Photograph estimates; real intrinsics, distortion and extrinsics remain unmeasured',
                evidence=profile['evidence'])
