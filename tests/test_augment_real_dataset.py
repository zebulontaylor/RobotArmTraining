"""Checks of tools/augment_real_dataset.py that need no models, GPU or network."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from tools import augment_real_dataset as aug


def wood_table_scene(size=128, square=(40, 40, 88, 88)):
    """A wood-coloured table with a grey object standing on it, and a matching disparity map."""
    image = np.zeros((size, size, 3), np.uint8)
    image[:] = (60, 140, 200)  # BGR wood
    x0, y0, x1, y1 = square
    image[y0:y1, x0:x1] = (128, 128, 128)
    yy = np.mgrid[0:size, 0:size][0].astype(np.float32)
    disparity = .3 + .4 * yy / size  # the table tilts away from the camera
    disparity[y0:y1, x0:x1] += .3  # the object rises above it
    return image, disparity, square


class MaskTest(unittest.TestCase):
    def test_keeps_the_object_and_repaints_the_table(self):
        image, disparity, (x0, y0, x1, y1) = wood_table_scene()
        for clean in (True, False):
            with self.subTest(clean=clean):
                keep = aug.keep_mask(image, disparity, clean=clean)
                table = np.ones_like(keep)
                table[y0 - 12:y1 + 12, x0 - 12:x1 + 12] = False
                self.assertGreater(keep[y0 + 4:y1 - 4, x0 + 4:x1 - 4].mean(), .95)
                self.assertLess(keep[table].mean(), .05)

    def test_scramble_never_touches_kept_pixels(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (3, 64, 64, 3), dtype=np.uint8)
        keep = np.zeros((3, 64, 64), bool)
        keep[:, 20:40, 10:30] = True
        for realistic in (True, False):
            with self.subTest(realistic=realistic):
                out, kind = aug.scramble_background(frames, keep, seed=5, realistic=realistic)
                self.assertTrue(np.array_equal(out[keep], frames[keep]))
                self.assertFalse(np.array_equal(out[~keep], frames[~keep]))
                self.assertTrue(kind.startswith('realistic' if realistic else 'wild'))

    def test_maps_round_trip(self):
        rng = np.random.default_rng(1)
        keep = rng.random((5, 64, 64)) > .5
        disparity = cv2.GaussianBlur(rng.random((5, 64, 64)).astype(np.float32), (0, 0), 4)
        with tempfile.TemporaryDirectory() as tmp:
            aug.save_maps(Path(tmp) / 'maps.npz', {camera: (keep, disparity) for camera in aug.CAMERAS})
            loaded = aug.load_maps(Path(tmp) / 'maps.npz')
        for camera in aug.CAMERAS:
            self.assertTrue(np.array_equal(loaded[camera][0], keep))  # masks are stored exactly
            self.assertEqual(loaded[camera][1].shape, disparity.shape)


class RelightTest(unittest.TestCase):
    def test_still_scene_stays_still_when_depth_scale_jitters(self):
        """Monocular depth has its own scale and offset in every frame; a still scene must not flicker."""
        rng = np.random.default_rng(2)
        frame = cv2.GaussianBlur(rng.integers(60, 200, (96, 96, 3), dtype=np.uint8), (0, 0), 3)
        frames = np.repeat(frame[None], 30, 0)
        yy, xx = np.mgrid[0:96, 0:96].astype(np.float32)
        base = yy / 96 + .2 * np.sin(xx / 9)
        disparity = np.stack([base * rng.uniform(.6, 1.4) + rng.uniform(-.3, .3) for _ in range(30)]).astype(np.float32)
        look = aug.sample_lighting(3)
        look.update(shadows=0)  # drifting shadow blobs move on purpose
        out = aug.relight(frames, disparity, look, camera='overhead').astype(np.float32).mean(-1)
        jumps = np.abs(np.diff(out.mean((1, 2))))
        # Out of 255; normalising depth once per episode instead of per frame gave 0.38 here.
        self.assertLess(np.median(jumps[5:]), .1)


class ClaimTest(unittest.TestCase):
    def test_two_pods_share_a_work_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            episode = work / 'episode_0001_restyled0'
            as_pod = lambda name: mock.patch.object(socket, 'gethostname', return_value=name)
            with as_pod('pod-a'):
                self.assertTrue(aug.claim(episode))
            with as_pod('pod-b'):
                self.assertFalse(aug.claim(episode))  # pod-a is rendering it
            with as_pod('pod-a'):
                self.assertTrue(aug.claim(episode))  # pod-a restarted: its own claim
            episode.mkdir()
            (episode / 'done.json').write_text('{}')
            with as_pod('pod-b'):
                self.assertFalse(aug.claim(episode))  # finished
            stale = work / 'episode_0002_restyled0'
            with as_pod('pod-a'):
                self.assertTrue(aug.claim(stale))
            old = time.time() - aug.CLAIM_STALE_S - 10
            os.utime(work / (stale.name + '.claim'), (old, old))
            with as_pod('pod-b'):
                self.assertTrue(aug.claim(stale))  # abandoned claims are taken over

    def test_a_live_process_on_the_same_host_keeps_its_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = Path(tmp) / 'episode_0001_restyled0'
            mark = Path(tmp) / (episode.name + '.claim')
            mark.write_text(f'{socket.gethostname()}:{os.getppid()}')  # our parent process: alive
            self.assertFalse(aug.claim(episode))
            dead = subprocess.Popen([sys.executable, '-c', 'pass'])
            dead.wait()
            mark.write_text(f'{socket.gethostname()}:{dead.pid}')  # a finished process: abandoned
            self.assertTrue(aug.claim(episode))
            self.assertEqual(mark.read_text(), f'{socket.gethostname()}:{os.getpid()}')


class RerunTest(unittest.TestCase):
    def test_settings_record_refuses_other_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            aug.check_settings(Path(tmp), dict(seed=1, source='a'))
            aug.check_settings(Path(tmp), dict(seed=1, source='a'))  # the same run again is fine
            with self.assertRaises(SystemExit):
                aug.check_settings(Path(tmp), dict(seed=99, source='a'))

    def test_stripped_folders_are_rendered_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'episode_0000_lit0'
            for camera in aug.CAMERAS:
                (dest / camera).mkdir(parents=True)
                (dest / camera / '00000.jpg').write_bytes(b'x')
            (dest / 'done.json').write_text(json.dumps(dict(frames=1)))
            self.assertTrue(aug.episode_ready(dest))
            self.assertEqual(aug.reopen_stripped([dest]), [])
            for camera in aug.CAMERAS:  # what a dataset write without --keep-jpegs leaves behind
                (dest / camera / '00000.jpg').unlink()
            self.assertFalse(aug.episode_ready(dest))
            self.assertEqual(aug.reopen_stripped([dest]), [dest])
            self.assertFalse((dest / 'done.json').exists())


def synthetic_source(root, size=16):
    """A minimal dataset description in the real format: meta/info.json and meta/tasks.parquet."""
    import pandas as pd
    vector = dict(dtype='float32', shape=[14], names=None)
    features = {k: dict(vector) for k in aug.NUMERIC}
    features.update({f'observation.images.{c}': dict(dtype='image', shape=[size, size, 3],
                                                      names=['height', 'width', 'channels']) for c in aug.CAMERAS})
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps(dict(fps=20, robot_type='test', features=features)))
    pd.DataFrame({'task_index': [0]}, index=['put the gear in the carrier']).to_parquet(root / 'meta/tasks.parquet')


def synthetic_episode(root, dest, frames=3, size=16):
    rng = np.random.default_rng(len(dest.name))
    numeric = {k: rng.random((frames, 14)).astype(np.float32) for k in aug.NUMERIC}
    images = {c: rng.integers(0, 255, (frames, size, size, 3), dtype=np.uint8) for c in aug.CAMERAS}
    dest.mkdir(parents=True)
    aug.write_folder(root, dest, numeric, images, dict(frames=frames))


@unittest.skipUnless(__import__('importlib').util.find_spec('lerobot'), 'needs lerobot')
class WriteDatasetTest(unittest.TestCase):
    def test_rerun_after_cleanup_keeps_the_finished_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            synthetic_source(tmp / 'source')
            dests = [tmp / 'work' / f'episode_000{i}' for i in range(2)]
            for dest in dests:
                synthetic_episode(tmp / 'source', dest)
            output = tmp / 'out'
            aug.write_dataset(tmp / 'source', output, dests, dict(test=True))  # removes the JPEGs
            self.assertTrue((output / 'COMPLETE').exists())
            self.assertTrue(aug.output_matches(output, dests))
            with self.assertRaises(SystemExit):  # the folders lost their images: refuse, change nothing
                aug.write_dataset(tmp / 'source', output, dests, dict(test=True))
            self.assertTrue((output / 'COMPLETE').exists())
            self.assertTrue((output / 'meta/provenance.json').exists())


if __name__ == '__main__':
    unittest.main()
