"""Check continuous collection boundaries and streaming without rendering."""
import numpy as np

from collection.continuous_pin import ContinuousConfig, collect_continuous


class Backend:
    hz = 10.
    open_width = .025
    closed_width = 0.

    def __init__(self, fail_cycle=None):
        self.resets = 0
        self.position = np.array([.2, 0., .1])
        self.rotation = np.eye(3)
        self.lifts = 0
        self.fail_cycle = fail_cycle

    def reset(self, seed):
        self.resets += 1
        return {'approach_retreat_direction': [0., 0., 1.]}

    def observe(self):
        return dict(ee_position=self.position, ee_rotation=self.rotation,
                    image_overview=np.zeros((2, 2, 3), dtype=np.uint8))

    def command(self, position, rotation, opening):
        self.position, self.rotation = position.copy(), rotation.copy()

    def grasped(self):
        return True

    def supported(self):
        return True

    def verify_lift(self, height):
        if self.fail_cycle is not None and self.lifts >= self.fail_cycle*3:
            return False
        self.lifts += 1
        return True

    def stop(self):
        self.stopped = True


def test_single_reset_streams_every_step_and_bounds_velocity():
    backend = Backend()
    frames = []
    config = ContinuousConfig()
    arrays, result = collect_continuous(backend, 0, 3, config, lambda o, p, c: frames.append((p, c)))
    assert result['success'], result
    assert backend.resets == 1 and backend.stopped
    assert len(frames) == len(arrays['actions']) + 1
    assert not any(k.startswith('obs_image') for k in arrays)
    assert result['completed_cycles'] == 3
    assert set(arrays['cycle_ids']) == {0, 1, 2}
    velocity = np.linalg.norm(np.diff(arrays['actions'][:, :3], axis=0), axis=1)*backend.hz
    assert velocity.max() <= config.speed_m_s*1.001
    fingers = abs(np.diff(arrays['actions'][:, -1]))*backend.hz
    assert fingers.max() <= config.opening_speed_m_s*1.001


def test_failure_preserves_prior_cycle_labels_without_reset():
    backend = Backend(fail_cycle=1)
    arrays, result = collect_continuous(backend, 2, 3)
    assert not result['success']
    assert result['completed_cycles'] == 1
    assert backend.resets == 1
    assert arrays['training_mask'][arrays['cycle_ids'] == 0].any()
    assert not arrays['training_mask'][arrays['cycle_ids'] == 1].any()


def test_fingers_close_during_approach_with_clearance_until_arrival():
    backend = Backend()
    config = ContinuousConfig()
    arrays, result = collect_continuous(backend, 0, 2, config)
    assert result['success']
    for cycle in range(2):
        returning = arrays['actions'][(arrays['cycle_ids'] == cycle) & (arrays['phases'] == 2)]
        widths = returning[:, -1]
        assert np.isclose(widths[0], result['cycles'][cycle]['approach_start_opening_m'])
        assert np.all(np.diff(widths) <= 1e-12)
        assert np.isclose(widths[-1], result['cycles'][cycle]['approach_arrival_opening_m'])
        remaining = np.linalg.norm(returning[:, :3]-returning[-1, :3], axis=1)
        assert np.any((remaining > .005) & (widths < widths[0]-.001))
        assert widths.min() > backend.closed_width
    starts = [r['approach_start_opening_m'] for r in result['cycles']]
    assert np.ptp(starts) > .001
    assert all(config.approach_start_min_m <= w <= config.approach_start_max_m for w in starts)
