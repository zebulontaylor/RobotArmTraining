"""Reset distributions, physical stepping and RL episode boundaries."""
from dataclasses import replace

from gymnasium.utils.env_checker import check_env
import mujoco
import numpy as np
import pytest

from sim.actuator_env import TABLE_Z
from sim.pin_insertion_env import BimanualPinInsertionEnv, PinInsertionConfig, rotation
from tools.try_actuator_assembly import pin_placement


@pytest.fixture(scope='module')
def env():
    result = BimanualPinInsertionEnv()
    yield result
    result.close()


def test_gymnasium_api_and_determinism(env):
    check_env(env, skip_render_check=True)


def test_seed_repeats_state_and_unseeded_reset_advances(env):
    obs, info = env.reset(seed=91)
    first = env.data.qpos.copy()
    params = info['reset']
    env.step(np.zeros(14))
    repeated, info = env.reset(seed=91)
    np.testing.assert_array_equal(env.data.qpos, first)
    for key in obs:
        np.testing.assert_array_equal(repeated[key], obs[key])
    np.testing.assert_array_equal(params['pin_grip_rotation_deg'], info['reset']['pin_grip_rotation_deg'])
    env.reset()
    assert not np.array_equal(first, env.data.qpos)


def test_randomized_resets_are_grasped_above_hole_with_independent_offsets(env):
    positions, holders, carriers = [], [], []
    for seed in range(8):
        obs, info = env.reset(seed=seed)
        p = info['reset']
        assert env.observation_space.contains(obs)
        assert info['pinched'] and info['carrier_supported'] and info['holder_pinched']
        assert not info['is_success']
        assert np.max(abs(p['arm_offset_m'][:2])) <= .005
        assert .020 <= p['arm_offset_m'][2] <= .040
        assert np.max(abs(p['pin_grip_translation_m'])) <= .001
        assert p['pin_grip_translation_m'][1] == 0
        assert np.max(abs(p['pin_grip_rotation_deg'])) <= 1
        assert np.max(abs(p['holder_translation_m'])) <= .0005
        assert np.max(abs(p['holder_rotation_deg'])) <= 2
        assert np.max(abs(p['carrier_offset_m'][:2])) <= .030
        assert p['carrier_offset_m'][2] == 0
        assert abs(p['carrier_yaw_deg']) <= 10
        fixture = env.model.body('small_carrier').id
        np.testing.assert_allclose(env.data.xpos[fixture],
                                   np.r_[env.config.carrier_center_xy_m, TABLE_Z + .0002]
                                   + p['carrier_offset_m'], atol=1e-12)
        np.testing.assert_allclose(env.data.xmat[fixture].reshape(3, 3),
                                   rotation([0, 0, np.deg2rad(-90 + p['carrier_yaw_deg'])]),
                                   atol=1e-12)
        assert info['axial_error_mm'] > 30
        assert all(c.dist >= -.0005 for c in env.data.contact)
        positions.append(p['actual_pin_in_grip_m'])
        holders.append(env.data.site_xmat[env.sim.arms['right'].ee_site].copy())
        carriers.append(np.r_[p['carrier_offset_m'][:2], p['carrier_yaw_deg']])
    assert np.ptp(positions, axis=0).max() > .001
    assert np.ptp(holders, axis=0).max() > .01
    assert np.all(np.ptp(carriers, axis=0) > [.001, .001, 1.])


def test_nominal_reset_disables_carrier_and_grip_randomization(env):
    _, info = env.reset(seed=17, options={'randomize': False})
    p = info['reset']
    assert p['carrier_yaw_deg'] == 0
    for name in ('carrier_offset_m', 'pin_grip_translation_m', 'pin_grip_rotation_deg',
                 'holder_translation_m', 'holder_rotation_deg'):
        np.testing.assert_array_equal(p[name], np.zeros(3))
    assert info['holder_pinched'] and info['pinched']
    np.testing.assert_allclose(env.carrier_reference[:2], [.26, 0], atol=1e-12)


def test_task_observation_exposes_actual_gear_bore(env):
    from tools.try_actuator_assembly import pin_target_pose
    obs, _ = env.reset(seed=0)
    gear_r = env.data.xmat[env.gear_id].reshape(3, 3)
    parts = env.sim.manifest['parts']
    offset = np.asarray(parts[env.pin_name]['assembled_origin']) - parts['gear_3']['assembled_origin']
    gear_target = env.data.xpos[env.gear_id] + gear_r @ offset
    carrier_target, _ = pin_target_pose(env.sim, env.pin_name)
    assert obs['task'].shape == (33,)
    np.testing.assert_allclose(obs['task'][-6:-3], gear_target - carrier_target, atol=1e-9)
    np.testing.assert_allclose(obs['task'][-3:], gear_r[:, 2], atol=1e-7)
    assert np.linalg.norm(obs['task'][-6:-3]) > 0


@pytest.mark.parametrize('x,y,yaw', [(x, y, yaw) for x in (-.03, .03)
                                   for y in (-.03, .03) for yaw in (-10., 10.)])
def test_carrier_distribution_corners_remain_reachable(env, monkeypatch, x, y, yaw):
    rng = np.random.default_rng(17)
    class CornerRng:
        def uniform(self, low, high, size=None):
            if np.isscalar(low) and low == -.03 and high == .03 and size == 2:
                return np.array([x, y])
            if np.isscalar(low) and low == -10 and high == 10 and size is None:
                return yaw
            return rng.uniform(low, high, size)

        def integers(self, *args, **kwargs):
            return rng.integers(*args, **kwargs)

    monkeypatch.setattr(env, '_np_random', CornerRng())
    _, info = env.reset()
    np.testing.assert_array_equal(info['reset']['carrier_offset_m'], [x, y, 0])
    assert info['reset']['carrier_yaw_deg'] == yaw
    assert info['holder_pinched'] and info['pinched'] and info['carrier_supported']


def test_bases_share_table_edge_with_colliding_tower_between_them(env):
    env.reset(seed=0)
    model = env.model
    left = model.body('left_base').id
    right = model.body('right_base').id
    tower = model.body('center_tower').id
    np.testing.assert_allclose(env.data.xmat[left], env.data.xmat[right])
    assert env.data.xpos[left, 0] == env.data.xpos[right, 0]
    assert env.data.xpos[left, 2] == env.data.xpos[right, 2]
    assert env.data.xpos[right, 1] < env.data.xpos[tower, 1] < env.data.xpos[left, 1]
    column = model.geom('tower_column').id
    arm_geom = model.geom('L_finger_pad').id
    assert model.geom_contype[column] & model.geom_conaffinity[arm_geom]
    assert model.ncam == 3 and model.neq == 2
    for arm in env.sim.arms.values():
        assert model.site_rgba[arm.ee_site, 3] == 0


def test_recorded_orientations_survive_base_frame_mapping(env):
    env.reset(seed=0)
    for sample in env.profile['samples']:
        for side in ('left', 'right'):
            original = np.array(sample[side]['rotation_base'])
            aligned = env._aligned_rotation(sample, side)
            np.testing.assert_allclose(aligned, original, atol=1e-12)
            np.testing.assert_allclose(aligned.T @ aligned, np.eye(3), atol=1e-10)


def test_step_uses_physics_and_retains_pin_without_grasp_constraints(env, monkeypatch):
    env.reset(seed=2)
    assert env.model.neq == 2  # Finger couplings only.
    assert env.model.jnt_type[env.model.joint('pin_2').id] == mujoco.mjtJoint.mjJNT_FREE
    start = env.data.time
    def forbid_teleport(*args, **kwargs):
        raise AssertionError('Part poses cannot be assigned during a rollout')
    monkeypatch.setattr(env.sim, 'set_part_pose', forbid_teleport)
    action = np.zeros(14)
    action[0] = .2
    previous = np.concatenate(list(env.commands.values()))
    for _ in range(8):
        obs, reward, end, timeout, info = env.step(action)
        current = np.concatenate(list(env.commands.values()))
        assert np.max(abs(current - previous)) <= .6 * .05 + 1e-12
        previous = current
        assert np.isfinite(reward) and env.observation_space.contains(obs)
        assert not end and not timeout
    assert env.data.time == pytest.approx(start + .4)
    assert info['pinched'] and info['carrier_supported'] and info['holder_pinched']


def test_timeout_is_truncation_and_requires_reset(env):
    original = env.config
    try:
        env.config = replace(original, max_steps=2)
        env.reset(seed=5)
        assert env.step(np.zeros(14))[2:4] == (False, False)
        _, _, end, timeout, info = env.step(np.zeros(14))
        assert not end and timeout and not info['is_success']
        with pytest.raises(RuntimeError, match='reset'):
            env.step(np.zeros(14))
    finally:
        env.config = original


def test_stationary_holder_retains_carrier_over_four_seconds(env):
    env.reset(seed=0)
    for _ in range(80):
        _, _, end, timeout, info = env.step(np.zeros(14))
        assert not end and not timeout
        assert info['holder_pinched'] and info['carrier_supported']


def test_contact_overload_ends_episode(env, monkeypatch):
    env.reset(seed=4)
    monkeypatch.setattr(env, '_contact_load', lambda: 36.)
    _, reward, end, timeout, info = env.step(np.zeros(14))
    assert end and not timeout and reward < -4
    assert info['failure'] == 'excess_contact_force'
    assert not info['is_success']


def test_success_checks_both_bores_and_rejects_pin_on_carrier_face(env):
    env.reset(seed=3)
    s = env.sim
    s.reset(randomize=False)
    for name in ('large_carrier', 'gear_1', 'gear_2', 'gear_3', 'gear_4', 'pin_2'):
        s.set_part_pose(name, *s.assembled_pose(name))
    mujoco.mj_forward(s.model, s.data)
    assert pin_placement(s, 'pin_2')['success']
    target, quat = s.assembled_pose('pin_2')
    for delta in ([0, 0, .017], [.001, 0, 0]):
        s.set_part_pose('pin_2', target + delta, quat)
        mujoco.mj_forward(s.model, s.data)
        assert not pin_placement(s, 'pin_2')['success']


@pytest.mark.parametrize('action', [np.zeros(13), np.full(14, np.nan), np.full(14, np.inf)])
def test_invalid_actions_do_not_advance_physics(env, action):
    env.reset(seed=0)
    start = env.data.time
    with pytest.raises(ValueError):
        env.step(action)
    assert env.data.time == start


@pytest.mark.parametrize('kwargs', [dict(start_xy_m=-.1), dict(start_height_m=(.04, .02)),
                                   dict(max_steps=0), dict(contact_limit_n=float('nan'))])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PinInsertionConfig(**kwargs)
