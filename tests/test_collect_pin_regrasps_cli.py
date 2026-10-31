"""Check dataset orchestration with serial, simulator-free attempts."""
import json

import pytest

from tools import collect_pin_regrasps as cli


def fake_attempts(monkeypatch, outcomes):
    requests = []
    outcomes = iter(outcomes)

    def record(request):
        requests.append(request)
        success = next(outcomes)
        return dict(seed=request[0], success=success,
                    failure=None if success else 'missed grasp',
                    training_steps=7 if success else 0)

    monkeypatch.setattr(cli, 'record_attempt', record)
    return requests


def read_json(path):
    return json.loads(path.read_text())


def test_stops_at_requested_successes_and_writes_dataset_contract(tmp_path, monkeypatch):
    requests = fake_attempts(monkeypatch, [False, True, True])
    config_path = tmp_path / 'overrides.json'
    config_path.write_text(json.dumps({'lift_m': 0.04, 'close_s': 0.8}))
    output = tmp_path / 'dataset'
    assert cli.main(['--output', str(output), '--episodes', '2', '--max-attempts', '5',
                     '--seed', '19', '--config', str(config_path), '--hz', '20',
                     '--image-size', '64', '--no-images', '--save-failures']) == 0
    assert [r[0] for r in requests] == [19, 20, 21]
    assert all(r[1] == str(output) and r[4] for r in requests)
    assert requests[0][2]['lift_m'] == 0.04
    config = read_json(output / 'config.json')
    assert config['collection']['close_s'] == 0.8
    assert config['backend_config'] == {'hz': 20, 'image_size': 64, 'images': False}
    assert len(config['action_fields']) == 13
    assert config['observation_alignment'] == 'obs[i], actions[i], obs[i+1]'
    assert config['seed'] == 19
    manifest = [json.loads(line) for line in (output / 'attempts.jsonl').read_text().splitlines()]
    assert [r['seed'] for r in manifest] == [19, 20, 21]
    assert [r['success'] for r in manifest] == [False, True, True]
    summary = read_json(output / 'summary.json')
    assert summary['status'] == 'complete'
    assert summary['attempts'] == 3
    assert summary['successful_episodes'] == summary['requested_episodes'] == 2
    assert summary['training_transitions'] == 14


def test_attempt_limit_returns_failure_and_preserves_manifest(tmp_path, monkeypatch):
    requests = fake_attempts(monkeypatch, [False, True, False, False])
    assert cli.main(['--output', str(tmp_path), '--episodes', '2', '--max-attempts', '4']) == 1
    assert len(requests) == 4
    summary = read_json(tmp_path / 'summary.json')
    assert summary['status'] == 'attempt_limit'
    assert summary['attempts'] == 4
    assert summary['successful_episodes'] == 1
    assert summary['training_transitions'] == 7
    assert len((tmp_path / 'attempts.jsonl').read_text().splitlines()) == 4


def test_seed_sequence_is_unique_and_reproducible(tmp_path, monkeypatch):
    runs = []
    for name in ('first', 'second'):
        requests = fake_attempts(monkeypatch, [False, False, True])
        assert cli.main(['--output', str(tmp_path / name), '--episodes', '1',
                         '--max-attempts', '3', '--seed', '71']) == 0
        runs.append([r[0] for r in requests])
    assert runs[0] == runs[1] == [71, 72, 73]
    assert len(set(runs[0])) == len(runs[0])


def test_nonempty_output_is_rejected_without_changes(tmp_path, monkeypatch):
    existing = tmp_path / 'config.json'
    existing.write_text('original dataset')
    requests = fake_attempts(monkeypatch, [])
    with pytest.raises(SystemExit) as error:
        cli.main(['--output', str(tmp_path)])
    assert error.value.code == 2
    assert existing.read_text() == 'original dataset'
    assert not requests
    assert sorted(p.name for p in tmp_path.iterdir()) == ['config.json']


def test_attempt_exception_is_reported_and_propagated(tmp_path, monkeypatch):
    def fail(request):
        raise RuntimeError('controller unavailable')

    monkeypatch.setattr(cli, 'record_attempt', fail)
    with pytest.raises(RuntimeError, match='controller unavailable'):
        cli.main(['--output', str(tmp_path), '--episodes', '1'])
    summary = read_json(tmp_path / 'summary.json')
    assert summary['status'] == 'interrupted_or_error'
    assert summary['successful_episodes'] == 0
