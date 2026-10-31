"""Check bounded video streams and observation timing with real H.264 encoding."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from tools.collect_continuous_pin import CAMERAS, VideoStreams


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='Requires ffmpeg')
def test_camera_streams_match_all_observations(tmp_path):
    streams = VideoStreams(tmp_path, 30)
    # Two actions have three observations, including the initial and final states.
    for index in range(3):
        observation = {'time': 10 + index/30}
        for camera in CAMERAS:
            observation['image_'+camera] = np.full((17, 19, 3), [220, 30, 10], np.uint8)
        streams.record(observation, np.zeros((32, 48, 3), np.uint8), 'close', 0)
    streams.close()
    manifest = [json.loads(line) for line in (tmp_path/'frames.jsonl').read_text().splitlines()]
    assert [row['observation_index'] for row in manifest] == [0, 1, 2]
    assert [row['time'] for row in manifest] == [10, 10+1/30, 10+2/30]
    for name in ('continuous', *CAMERAS):
        metadata = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(tmp_path/(name+'.mp4'))]))['streams'][0]
        assert int(metadata['nb_frames']) == 3
        assert metadata['r_frame_rate'] == '30/1'
        assert (metadata['width'], metadata['height']) == ((48, 32) if name == 'continuous' else (19, 17))
    decoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(tmp_path/'overview.mp4'),
                                      '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    pixels = np.frombuffer(decoded, np.uint8).reshape(3, 17, 19, 3)
    assert np.max(np.abs(pixels.astype(int) - [220, 30, 10])) < 8


def test_encoder_failure_preserves_log_and_closes_streams(tmp_path, monkeypatch):
    streams = VideoStreams(tmp_path, 30)
    monkeypatch.setattr('tools.collect_continuous_pin.subprocess.Popen',
                        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError('ffmpeg')))
    with pytest.raises(FileNotFoundError):
        streams.write('overview', np.zeros((16, 16, 3), np.uint8), 'rgb24')
    streams.close(abort=True)
    assert streams.manifest.closed
    assert (tmp_path/'overview.ffmpeg.log').exists()


def test_reset_failure_saves_summary_without_observations(tmp_path, monkeypatch):
    from collection import continuous_pin
    from sim import pin_regrasp_backend
    from tools import collect_continuous_pin as cli

    class Backend:
        hz = 30

        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(pin_regrasp_backend, 'MujocoPinBackend', Backend)
    monkeypatch.setattr(continuous_pin, 'collect_continuous', lambda *args, **kwargs:
                        ({'actions': np.empty((0, 13))},
                         {'success': False, 'failure': 'Reset failed', 'completed_cycles': 0}))
    output = tmp_path/'run'
    assert cli.main(['--output', str(output)]) == 1
    summary = json.loads((output/'summary.json').read_text())
    assert summary['failure'] == 'Reset failed'
    assert summary['video_frames'] == 0
    assert not (output/'error.json').exists()
    assert not (output/'continuous.mp4').exists()
