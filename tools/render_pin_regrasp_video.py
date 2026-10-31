"""Render recorded regrasp camera observations at their recorded playback speed.

Run from the repository root, for example::

    python tools/render_pin_regrasp_video.py --input data/regrasps --output regrasp.mp4

The input can be a collection root or its episodes/failures directory. Both
categories are included when present; --limit selects the first paths in order.
The adjacent .manifest.json records the selected inputs and playback timing.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


WIDTH, HEIGHT = 1280, 720
BACKGROUND = (24, 20, 16)
INK = (236, 237, 233)
ACCENT = (215, 195, 80)


def nearest_indices(times, fps):
    """Sample the nearest observation, preserving the final recorded state."""
    times = np.asarray(times, dtype=float).reshape(-1)
    if len(times) < 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('obs_time must contain finite, strictly increasing timestamps')
    times = times - times[0]
    count = max(1, math.ceil(times[-1]*fps-1e-9)+1)
    targets = np.minimum(np.arange(count)/fps, times[-1])
    right = np.searchsorted(times, targets).clip(0, len(times)-1)
    left = np.maximum(right-1, 0)
    return np.where(targets-times[left] <= times[right]-targets+1e-12, left, right), times


def label(canvas, text, xy, scale=0.6, color=INK):
    cv2.putText(canvas, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def panel(canvas, image, rect, title):
    x, y, w, h = rect
    canvas[y:y+h, x:x+w] = (9, 9, 9)
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Camera observations must be uint8 RGB images')
    scale = min(w/image.shape[1], (h-32)/image.shape[0])
    iw, ih = max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))
    resized = cv2.resize(image[..., ::-1], (iw, ih), interpolation=cv2.INTER_LINEAR)
    ox, oy = x+(w-iw)//2, y+32+(h-32-ih)//2
    canvas[oy:oy+ih, ox:ox+iw] = resized
    label(canvas, title, (x+12, y+22), 0.52, ACCENT)


def render_frame(cameras, index, elapsed, duration, phase, seed, scene, outcome=None):
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
    label(canvas, 'PIN REGRASP / TWO-ARM SIMULATION', (24, 34), 0.75)
    label(canvas, f'{scene}  |  seed {seed}', (24, 62), 0.53, ACCENT)
    main = next((key for key in ('overview', 'overhead') if key in cameras), next(iter(cameras)))
    wrists = [key for key in ('left_wrist', 'right_wrist') if key in cameras and key != main]
    if wrists:
        panel(canvas, cameras[main][index], (24, 82, 832, 560), main.replace('_', ' ').upper())
        for slot, key in enumerate(wrists):
            panel(canvas, cameras[key][index], (872, 82+slot*288, 384, 272), key.replace('_', ' ').upper())
    else:
        panel(canvas, cameras[main][index], (24, 82, 1232, 560), main.replace('_', ' ').upper())
    label(canvas, f'{phase.upper()}   {elapsed:5.2f} / {duration:.2f} s', (24, 682), 0.7)
    if outcome:
        label(canvas, outcome, (640, 682), 0.55, (125, 220, 145) if outcome.startswith('VERIFIED') else (130, 155, 245))
    return canvas


def render(input_dir, output, limit=3, fps=30, overwrite=False):
    output = Path(output)
    manifest_path = output.with_suffix('.manifest.json')
    if not overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError(f'Output already exists: {output} or {manifest_path}')
    paths = sorted(Path(input_dir).rglob('episode_*.npz'))[:limit]
    if not paths:
        raise ValueError(f'No episode_*.npz files in {input_dir}')
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg is required to encode H.264 video')
    output.parent.mkdir(parents=True, exist_ok=True)
    records, total_frames = [], 0
    with tempfile.TemporaryDirectory(prefix='.regrasp-video-', dir=output.parent) as temporary:
        temporary = Path(temporary)
        video = temporary / 'video.mp4'
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                   '-pix_fmt', 'bgr24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(fps), '-i', '-',
                   '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt',
                   'yuv420p', '-movflags', '+faststart', str(video)]
        with (temporary/'ffmpeg.log').open('w+') as error:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error)
            try:
                for path in paths:
                    result = json.loads(path.with_suffix('.json').read_text())
                    with np.load(path, allow_pickle=False) as data:
                        cameras = {key.removeprefix('obs_image_'): data[key] for key in data.files
                                   if key.startswith('obs_image_')}
                        if not cameras or 'obs_time' not in data:
                            raise ValueError(f'{path}: camera observations and obs_time are required')
                        indices, times = nearest_indices(data['obs_time'], fps)
                        if any(len(images) != len(times) for images in cameras.values()):
                            raise ValueError(f'{path}: camera/time observation counts differ')
                        phases, names = data['phases'], data['phase_names']
                        if len(phases) != len(times)-1:
                            raise ValueError(f'{path}: expected T phases and T+1 observations')
                        scene = str((result.get('reset') or {}).get('scene', 'Real-data approach | left: pin | right: hold carrier'))
                        seed = result.get('seed', path.stem)
                        title = render_frame(cameras, 0, 0, times[-1], 'episode start', seed, scene)
                        for _ in range(fps):
                            process.stdin.write(title.tobytes())
                        for index in indices:
                            phase = str(names[int(phases[max(0, index-1)])]) if len(phases) else 'reset'
                            frame = render_frame(cameras, index, times[index], times[-1], phase, seed, scene)
                            process.stdin.write(frame.tobytes())
                        outcome = ('VERIFIED: sustained physical grasp' if result.get('success') is True
                                   else 'FAILED: see episode JSON' if result.get('success') is False
                                   else 'OUTCOME NOT RECORDED')
                        terminal = render_frame(cameras, len(times)-1, times[-1], times[-1],
                                                phase, seed, scene, outcome)
                        for _ in range(fps):
                            process.stdin.write(terminal.tobytes())
                        count = len(indices)+2*fps
                        total_frames += count
                        records.append(dict(input=str(path.resolve()), seed=seed,
                                            success=result.get('success'), failure=result.get('failure'),
                                            recorded_duration_s=float(times[-1]), frames=count,
                                            cameras=list(cameras)))
                process.stdin.close()
                status = process.wait()
                if status:
                    error.seek(0)
                    raise RuntimeError(f'ffmpeg exited {status}: {error.read()}')
            except BaseException:
                process.stdin.close()
                if process.poll() is None:
                    process.terminate()
                process.wait()
                raise
        manifest = dict(input=str(Path(input_dir).resolve()), fps=fps, frames=total_frames,
                        duration_s=total_frames/fps, title_hold_s=1, terminal_hold_s=1,
                        playback='nearest observation at recorded obs_time; chronological phases',
                        episodes=records)
        metadata = temporary/'manifest.json'
        metadata.write_text(json.dumps(manifest, indent=2)+'\n')
        video.replace(output)
        metadata.replace(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.limit < 1 or args.fps < 1:
        parser.error('--limit and --fps must be positive')
    manifest = render(args.input, args.output, args.limit, args.fps, args.overwrite)
    print(json.dumps(dict(output=str(args.output), duration_s=manifest['duration_s'],
                          episodes=len(manifest['episodes']))))


if __name__ == '__main__':
    main()
