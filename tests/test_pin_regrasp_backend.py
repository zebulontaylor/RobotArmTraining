"""Exercise the collector through loaded physical pad contacts and a free pin."""
import numpy as np
import pytest

pytest.importorskip('mujoco')

from collection.pin_regrasp import collect_episode
from sim.pin_regrasp_backend import MujocoPinBackend


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_physical_release_return_and_lift(seed):
    backend = MujocoPinBackend(images=False)
    try:
        arrays, result = collect_episode(backend, seed)
        assert result['success'], result['failure']
        assert backend.sim.layout == 'side_by_side'
        assert arrays['obs_state'].shape[1] == 28
        assert arrays['obs_right_grasped'].all()
        carrier = arrays['obs_carrier_pose'][:, :3]
        assert np.max(np.linalg.norm(carrier - carrier[0], axis=1)) < .002
        assert result['reset']['right_task'] == 'hold_large_carrier'
        assert np.asarray(result['reset']['approach_retreat_direction'])[2] > 0
        assert backend.verify_lift(.018)
        pose = arrays['obs_object_pose']
        assert pose[-1, 2] - pose[0, 2] > .018
        release = arrays['phases'] == 0
        assert not arrays['obs_grasped'][1:][release][-1]
        assert arrays['obs_grasped'][-1]
        assert len(arrays['obs_time']) == len(arrays['actions']) + 1
        assert np.diff(arrays['obs_time']).min() > 0
    finally:
        backend.close()


def test_reject_invalid_control_pose_before_execution():
    backend = MujocoPinBackend(images=False)
    try:
        backend.reset(0)
        observation = backend.observe()
        with pytest.raises(ValueError, match='orthonormal'):
            backend.command(observation['ee_position'], np.zeros((3, 3)), 0.)
        assert backend.observe()['time'] == observation['time']
    finally:
        backend.close()
