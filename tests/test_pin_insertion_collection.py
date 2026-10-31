"""The regrasp exporter records physically executed insertion transitions."""
import numpy as np
import pytest

pytest.importorskip('mujoco')

from collection.pin_insertion import InsertionCollectionConfig, collect_insertion_episode
from sim.pin_regrasp_backend import MujocoPinInsertionBackend


@pytest.mark.parametrize('settings', [dict(endpoint_clearance_m=0), dict(endpoint_clearance_m=.02),
                                     dict(scene_xy_m=float('nan')),
                                     dict(retreat_min_m=.002), dict(max_steps=0)])
def test_reject_invalid_insertion_collection_settings(settings):
    with pytest.raises(ValueError):
        InsertionCollectionConfig(**settings)


def test_repeated_alignment_returns_keep_grasps_and_never_enter(monkeypatch):
    backend = MujocoPinInsertionBackend(images=False)
    reset = backend.reset

    def reset_then_forbid_pose_assignment(seed):
        metadata = reset(seed)

        def forbid_teleport(*args, **kwargs):
            raise AssertionError('Only reset may assign part poses')

        monkeypatch.setattr(backend.sim, 'set_part_pose', forbid_teleport)
        return metadata

    monkeypatch.setattr(backend, 'reset', reset_then_forbid_pose_assignment)
    try:
        arrays, result = collect_insertion_episode(backend, 0)
        assert result['success'], result
        assert arrays['actions'].shape == (result['steps'], 14)
        assert np.all(np.abs(arrays['actions']) <= 1)
        assert len(arrays['obs_time']) == result['steps']+1
        assert np.all(np.diff(arrays['obs_time']) > 0)
        assert arrays['obs_insertion_controller'].shape == (result['steps']+1, 64)
        names = arrays['phase_names'][arrays['phases']]
        assert {'verify', 'retreat', 'carrier_drag', 'pin_relocate', 'return'} == set(names)
        np.testing.assert_array_equal(arrays['training_mask'], names == 'return')
        assert arrays['obs_grasped'].all()
        assert arrays['obs_right_grasped'].all()
        np.testing.assert_array_equal(arrays['actions'][:, [6, 9, 10, 11, 12, 13]], 0)
        np.testing.assert_array_equal(arrays['actions'][names == 'return', 7:], 0)
        assert arrays['obs_alignment_carrier_table_supported'].all()
        assert result['task'] == 'alignment'
        assert result['alignment_mode'] == 'alignment_only'
        assert result['completed_cycles'] == result['requested_cycles'] == 3
        assert set(arrays['cycle_ids']) == {0, 1, 2}
        assert np.all(arrays['obs_alignment_clearance_m'] > 0)
        assert .0007 <= arrays['obs_alignment_clearance_m'][-1] <= .0013
        assert names[0] == 'return'
        assert result['approach_style'] == 'pickup_toward_base_40deg'
        assert result['setup_steps'] > 0
        assert arrays['obs_alignment_clearance_m'][0] > .005
        initial = arrays['obs_ee_position'][0]
        final = arrays['obs_ee_position'][-1]
        assert np.linalg.norm((initial-final)[:2]) > .01
        assert initial[2]-final[2] > .005
        for cycle in result['cycles']:
            assert cycle['retreat']['clearance_m'] > .005
            offset = np.asarray(cycle['retreat_offset_m'])
            assert np.linalg.norm(offset[:2]) > .01
            assert offset[2] > .005
            actual_offset = np.asarray(cycle['actual_retreat_offset_m'])
            axis = np.asarray(result['approach_retreat_direction'])
            assert np.linalg.norm(actual_offset[:2]) > .01
            assert actual_offset[:2] @ (axis[:2]/np.linalg.norm(axis[:2])) > .01
            actual = arrays['obs_ee_position'][cycle['return_action_start']]
            actual_end = arrays['obs_ee_position'][cycle['action_end']]
            assert np.linalg.norm((actual-actual_end)[:2]) > .01
            assert cycle['carrier_max_approach_drift_m'] < .0005
            if cycle['cycle']:
                displacement = np.asarray(cycle['carrier_drag']['displacement_m'])
                assert np.linalg.norm(displacement[:2]) >= .009
                assert np.linalg.norm(np.asarray(cycle['carrier_drag']['after_position_m'])[:2]
                                      - cycle['carrier_drag']['target_xy_m']) < .001
                assert abs(displacement[2]) < .001
            else:
                assert cycle['carrier_drag'] is None
            for endpoint in cycle['endpoint_samples']:
                assert .0007 <= endpoint['clearance_m'] <= .0013
                assert endpoint['mouth_error_m'] < .00017
                assert endpoint['pin_grasped'] and endpoint['carrier_grasped']
        assert not result['validation']['is_success']
        reset_metadata = result['reset']
        assert reset_metadata['pin_independent_xy_jitter_m'] == 0
        assert reset_metadata['carrier_on_table']
        assert reset_metadata['carrier_detail_camera']['fixed_world']
        np.testing.assert_array_equal(reset_metadata['carrier_detail_camera']['lookat_world_m'],
                                      backend._detail_lookat)
        assert reset_metadata['holder_grasp_height_m'] == .011
        np.testing.assert_array_equal(reset_metadata['arm_offset_m'][:2], 0)
        np.testing.assert_array_equal(reset_metadata['pin_grip_translation_m'], 0)
        np.testing.assert_array_equal(reset_metadata['holder_translation_m'], 0)
        np.testing.assert_array_equal(reset_metadata['holder_rotation_deg'], 0)
        np.testing.assert_array_equal(reset_metadata['shared_scene_offset_m'],
                                      reset_metadata['carrier_offset_m'])
        assert all(value.dtype.kind != 'O' for value in arrays.values())
    finally:
        backend.close()


def test_step_budget_rejects_attempt_without_training_labels():
    backend = MujocoPinInsertionBackend(images=False, max_steps=1)
    try:
        arrays, result = collect_insertion_episode(backend, 0,
                                                  InsertionCollectionConfig(max_steps=1))
        assert not result['success']
        assert result['failure']
        assert not arrays['training_mask'].any()
        assert len(arrays['obs_time']) == len(arrays['actions'])+1
    finally:
        backend.close()
