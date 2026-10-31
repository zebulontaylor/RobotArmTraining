"""Exercise collection contracts without a simulator or camera."""
import numpy as np
import pytest

from collection.pin_regrasp import (CollectionConfig, PHASE_NAMES, collect_episode,
                                    rotation_log, sample_retreat)


class FakeBackend:
    hz = 10.0
    open_width = 0.04
    closed_width = 0.005

    def __init__(self, *, grasp=True, lift=True, fail_command=None):
        self.final_grasp = grasp
        self.lift_ok = lift
        self.fail_command = fail_command
        self.commands = []
        self.stops = 0
        self.grasp_checks = 0
        self.tick = 0
        self.position = np.array([0.2, 0.0, 0.1])

    def reset(self, seed):
        return {'seed': seed}

    def observe(self):
        return {'ee_position': self.position, 'ee_rotation': np.eye(3),
                'image': np.full((2, 3, 3), self.tick, dtype=np.uint8)}

    def command(self, position, rotation, opening):
        if len(self.commands) == self.fail_command:
            raise RuntimeError('injected controller failure')
        self.commands.append(np.r_[position, rotation.ravel(), opening])
        self.tick += 1
        # Measured motion differs from commanded motion, making path replay
        # distinguishable from simply reversing the requested retreat.
        self.position = position + np.array([0.0002, 0.0001, 0.0])

    def grasped(self):
        self.grasp_checks += 1
        return self.grasp_checks == 1 or self.final_grasp

    def verify_lift(self, min_height):
        assert min_height > 0
        return self.lift_ok

    def stop(self):
        self.stops += 1


CONFIG = CollectionConfig(retreat_min_m=0.04, retreat_max_m=0.06,
                          speed_m_s=0.1, release_s=0.1, close_s=0.1,
                          verify_s=0.2)


def test_forward_observations_align_with_executed_actions():
    backend = FakeBackend()
    arrays, result = collect_episode(backend, 5, CONFIG)
    assert result['success'], result
    count = len(backend.commands)
    assert arrays['actions'].shape == (count, 13)
    np.testing.assert_allclose(arrays['actions'], backend.commands)
    assert arrays['obs_image'].shape[0] == count + 1
    np.testing.assert_array_equal(arrays['obs_image'][:, 0, 0, 0], np.arange(count + 1))
    np.testing.assert_allclose(arrays['obs_ee_position'][1:],
                               arrays['actions'][:, :3] + [0.0002, 0.0001, 0])
    expected = np.isin(arrays['phases'], [PHASE_NAMES.index('return'), PHASE_NAMES.index('close')])
    np.testing.assert_array_equal(arrays['training_mask'], expected)
    assert result['training_steps'] == expected.sum() > 0
    assert backend.stops == 1


def test_return_commands_retrace_measured_retreat_waypoints():
    arrays, result = collect_episode(FakeBackend(), 11, CONFIG)
    assert result['success']
    phases = arrays['phases']
    retreat_indices = np.flatnonzero(phases == PHASE_NAMES.index('retreat'))
    assert len(retreat_indices) > 1
    # Observation i+1 is the measured endpoint of action i. Include the
    # released anchor, but omit the last endpoint where the return starts.
    measured = arrays['obs_ee_position'][retreat_indices[0]:retreat_indices[-1] + 1]
    returned = arrays['actions'][phases == PHASE_NAMES.index('return'), :3]
    cursor = 0
    for target in measured[::-1]:
        matches = np.flatnonzero(np.all(np.isclose(returned[cursor:], target, atol=1e-12), axis=1))
        assert len(matches), f'Measured retreat target {target} was not replayed'
        cursor += matches[0] + 1


@pytest.mark.parametrize('kwargs, failure', [
    ({'grasp': False}, 'No bilateral grasp'),
    ({'lift': False}, 'sustained lift'),
    ({'fail_command': 3}, 'injected controller failure'),
])
def test_failed_episodes_have_no_training_labels_and_stop(kwargs, failure):
    backend = FakeBackend(**kwargs)
    arrays, result = collect_episode(backend, 4, CONFIG)
    assert not result['success']
    assert failure in result['failure']
    assert not arrays['training_mask'].any()
    assert result['training_steps'] == 0
    assert backend.stops == 1
    assert len(arrays['obs_image']) == len(arrays['actions']) + 1
    if 'fail_command' in kwargs:
        assert len(backend.commands) == kwargs['fail_command']


def test_seed_controls_retreat_variation():
    first, a = collect_episode(FakeBackend(), 21, CONFIG)
    second, b = collect_episode(FakeBackend(), 21, CONFIG)
    _, c = collect_episode(FakeBackend(), 22, CONFIG)
    np.testing.assert_array_equal(first['actions'], second['actions'])
    assert a['retreat_offset_m'] == b['retreat_offset_m']
    assert a['retreat_offset_m'] != c['retreat_offset_m']
    offset = np.asarray(a['retreat_offset_m'])
    assert CONFIG.retreat_min_m <= np.linalg.norm(offset) <= CONFIG.retreat_max_m
    assert offset[2] / np.linalg.norm(offset) >= 0.25


@pytest.mark.parametrize('field', list(CollectionConfig.__dataclass_fields__))
@pytest.mark.parametrize('value', [0.0, -1.0, np.inf, np.nan])
def test_invalid_config_values_are_rejected(field, value):
    with pytest.raises(ValueError):
        CollectionConfig(**{field: value})


def test_reversed_retreat_bounds_are_rejected():
    with pytest.raises(ValueError, match='retreat_min_m'):
        CollectionConfig(retreat_min_m=0.1, retreat_max_m=0.02)


def test_profile_direction_constrains_retreat_cone():
    axis = np.array([-.6, .3, .7])
    axis /= np.linalg.norm(axis)
    samples = [sample_retreat(np.random.default_rng(seed), CONFIG, axis) for seed in range(50)]
    for offset in samples:
        assert offset @ axis / np.linalg.norm(offset) >= np.cos(np.deg2rad(CONFIG.retreat_cone_deg)) - 1e-12
    assert np.ptp(samples, axis=0).min() > .001


def test_return_replays_orientation_and_respects_angular_rate():
    class OrientedBackend(FakeBackend):
        rotation = np.eye(3)

        def reset(self, seed):
            return dict(seed=seed, approach_retreat_direction=[-.3, .2, .9])

        def command(self, position, rotation, opening):
            super().command(position, rotation, opening)
            self.rotation = rotation.copy()

        def observe(self):
            return dict(super().observe(), ee_rotation=self.rotation.copy())

    backend = OrientedBackend()
    data, result = collect_episode(backend, 42, CONFIG)
    assert result['success']
    rotations = data['actions'][:, 3:12].reshape(-1, 3, 3)
    for first, second in zip(rotations, rotations[1:]):
        assert np.linalg.norm(rotation_log(first.T @ second)) <= np.deg2rad(CONFIG.angular_speed_deg_s)/backend.hz + 1e-10
    retreat = np.flatnonzero(data['phases'] == PHASE_NAMES.index('retreat'))
    assert np.linalg.norm(rotation_log(rotations[retreat[-1]])) > .001
    returned = rotations[data['phases'] == PHASE_NAMES.index('return')]
    measured = data['obs_ee_rotation'][retreat[0]:retreat[-1]+1]
    cursor = 0
    for target in measured[::-1]:
        matches = np.flatnonzero(np.all(np.isclose(returned[cursor:], target, atol=1e-10), axis=(1, 2)))
        assert len(matches)
        cursor += matches[0]+1
    np.testing.assert_allclose(returned[-1], np.eye(3), atol=1e-10)
