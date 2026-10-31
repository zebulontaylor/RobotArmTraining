"""Record one uninterrupted autonomous pin collection run and an H.264 video."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from tools.render_pin_regrasp_video import panel, label


def compose(observation, phase, cycle, cycles, elapsed):
    frame = np.full((720, 1280, 3), (24, 20, 16), np.uint8)
    label(frame, 'AUTONOMOUS PIN COLLECTION', (24, 34), .8)
    label(frame, 'Continuous simulation | left: pick and reposition | right: hold carrier',
          (24, 62), .53, (215, 195, 80))
    panel(frame, observation['image_overview'], (24, 82, 832, 560), 'TWO-ARM WORKSPACE')
    for slot, name in enumerate(('left_wrist', 'right_wrist')):
        panel(frame, observation['image_'+name], (872, 82+slot*288, 384, 272), name.replace('_', ' ').upper())
    label(frame, f'CYCLE {cycle+1}/{cycles}   {phase.replace("_", " ").upper()}   {elapsed:.1f} s',
          (24, 680), .65)
    holding = bool(observation.get('right_grasped', False))
    label(frame, 'Carrier: held' if holding else 'Carrier: contact lost', (930, 680), .57,
          (125, 220, 145) if holding else (130, 155, 245))
    return frame


CAMERAS = ('overview', 'overhead', 'left_wrist', 'right_wrist')


class VideoStreams:
    """Stream synchronized observations to encoders without buffering a run."""

    def __init__(self, directory, fps):
        self.directory, self.fps = Path(directory), fps
        self.encoders = {}
        self.frames = 0
        self.manifest = (self.directory/'frames.jsonl').open('w')

    def write(self, name, frame, pixel_format):
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Video frames must be uint8 RGB or BGR arrays')
        shape = frame.shape
        if name not in self.encoders:
            log = (self.directory/(name+'.ffmpeg.log')).open('w+')
            command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
                       '-pix_fmt', pixel_format, '-s', f'{shape[1]}x{shape[0]}',
                       '-r', str(self.fps), '-i', '-', '-an', '-c:v', 'libx264',
                       '-threads', '1', '-preset', 'fast', '-crf', '19',
                       '-pix_fmt', 'yuv420p' if name == 'continuous' else 'yuv444p', '-movflags', '+faststart',
                       str(self.directory/(name+'.mp4'))]
            try:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            except BaseException:
                log.close()
                raise
            self.encoders[name] = process, log, shape
        process, _, expected = self.encoders[name]
        if shape != expected:
            raise ValueError(f'Camera shape changed for {name}: {expected} to {shape}')
        process.stdin.write(frame.tobytes())

    def record(self, observation, composite, phase, cycle):
        self.write('continuous', composite, 'bgr24')
        for camera in CAMERAS:
            self.write(camera, observation['image_'+camera], 'rgb24')
        self.manifest.write(json.dumps(dict(frame=self.frames, observation_index=self.frames,
            time=float(observation['time']), video_time_s=self.frames/self.fps,
            phase=phase, cycle=cycle))+'\n')
        self.frames += 1

    def close(self, abort=False):
        errors = []
        self.manifest.close()
        for name, (process, log, _) in self.encoders.items():
            try:
                if not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                if abort and process.poll() is None:
                    process.terminate()
                try:
                    status = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    status = -1
                if status and not abort:
                    log.seek(0)
                    errors.append(f'{name}: encoder exited {status}: {log.read()}')
            finally:
                log.close()
        if errors:
            raise RuntimeError('; '.join(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=320)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args(argv)
    if min(args.cycles, args.image_size) < 1 or args.seed < 0:
        parser.error('Cycles and image size must be positive; seed must be nonnegative')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Choose an empty output directory')
    from collection.continuous_pin import ContinuousConfig, collect_continuous
    from sim.pin_regrasp_backend import MujocoPinBackend
    try:
        config = ContinuousConfig(**(json.loads(args.config.read_text()) if args.config else {}))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(dict(seed=args.seed, cycles=args.cycles,
        collection=asdict(config), image_size=args.image_size, initial_resets=1,
        video='All observed control frames, chronological, without cuts or time scaling',
        camera_videos={camera: camera+'.mp4' for camera in CAMERAS},
        frame_manifest='frames.jsonl', camera_encoding='Lossy H.264 CRF19, RGB input, YUV444 output'), indent=2)+'\n')
    backend = None
    try:
        backend = MujocoPinBackend(images=True, image_size=args.image_size)
        with tempfile.TemporaryDirectory(prefix='.video-', dir=args.output) as temporary:
            temporary = Path(temporary)
            streams = VideoStreams(temporary, backend.hz)
            started_at = None
            last_phase = None
            def record(observation, phase, cycle):
                nonlocal started_at, last_phase
                now = float(observation['time'])
                if started_at is None:
                    started_at = now
                if last_phase != (cycle, phase):
                    print(f'Cycle {cycle+1}/{args.cycles}: {phase} at {now-started_at:.1f}s', flush=True)
                    last_phase = (cycle, phase)
                frame = compose(observation, phase, cycle, args.cycles, now-started_at)
                streams.record(observation, frame, phase, cycle)
            try:
                arrays, result = collect_continuous(backend, args.seed, cycles=args.cycles,
                                                     config=config, on_step=record)
                streams.close()
                if len(arrays.get('obs_time', ())) != streams.frames:
                    raise RuntimeError('Video frames do not align with trajectory observations')
                np.savez_compressed(temporary/'trajectory.npz', **arrays)
                result.update(video_frames=streams.frames, video_fps=backend.hz,
                              video_duration_s=streams.frames/backend.hz,
                              camera_videos={camera: camera+'.mp4' for camera in CAMERAS},
                              frame_manifest='frames.jsonl')
                (temporary/'summary.json').write_text(json.dumps(result, indent=2, default=lambda x: x.tolist())+'\n')
            except BaseException:
                streams.close(abort=True)
                raise
            finally:
                for log in temporary.glob('*.ffmpeg.log'):
                    log.replace(args.output/log.name)
            for artifact in temporary.iterdir():
                artifact.replace(args.output/artifact.name)
            print(json.dumps(result, indent=2, default=lambda x: x.tolist()))
            return 0 if result['success'] else 1
    except BaseException as exc:
        (args.output/'error.json').write_text(json.dumps(dict(error=type(exc).__name__, message=str(exc)), indent=2)+'\n')
        raise
    finally:
        if backend is not None:
            backend.close()


if __name__ == '__main__':
    raise SystemExit(main())
