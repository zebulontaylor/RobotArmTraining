"""Collect randomized, physically verified pin regrasp demonstrations in MuJoCo."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from collection.pin_regrasp import CollectionConfig, PHASE_NAMES, collect_episode


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, default=json_default, indent=2) + '\n')
    temporary.replace(path)


def record_attempt(request):
    seed, output, config, backend_config, save_failures = request
    from sim.pin_regrasp_backend import MujocoPinBackend
    started = time.monotonic()
    backend = MujocoPinBackend(**backend_config)
    try:
        arrays, result = collect_episode(backend, seed, CollectionConfig(**config))
        result['elapsed_s'] = time.monotonic()-started
        if result['success'] or save_failures:
            category = 'episodes' if result['success'] else 'failures'
            path = Path(output) / category / f'episode_{seed:09d}.npz'
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(path)
            result['file'] = str(path.relative_to(output))
            write_json(path.with_suffix('.json'), result)
        return result
    finally:
        backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-attempts', type=int, default=500)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hz', type=int, default=30)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--no-images', action='store_true', help='State-only diagnostic collection')
    parser.add_argument('--save-failures', action='store_true')
    parser.add_argument('--config', type=Path, help='JSON overrides for CollectionConfig fields')
    args = parser.parse_args(argv)
    if min(args.episodes, args.max_attempts, args.workers, args.hz, args.image_size) < 1 or args.seed < 0:
        parser.error('Counts, frequency and image size must be positive; seed must be nonnegative')
    if args.max_attempts < args.episodes:
        parser.error('--max-attempts must be at least --episodes')
    if args.hz > 1000:
        parser.error('--hz cannot exceed the 1000 Hz physics timestep')
    try:
        config = CollectionConfig(**(json.loads(args.config.read_text()) if args.config else {}))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.output.exists() and not args.output.is_dir():
        parser.error('Output path must be a directory')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty; choose a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    backend_config = dict(hz=args.hz, image_size=args.image_size, images=not args.no_images)
    write_json(args.output / 'config.json', dict(
        schema_version=2, backend='MujocoPinBackend', backend_config=backend_config,
        setup_profile='photo_20261027', pose_profile='sim/actuator/pin_regrasp_profile.json',
        collection=asdict(config), seed=args.seed, requested_episodes=args.episodes,
        max_attempts=args.max_attempts, workers=args.workers,
        action_fields=['x_m', 'y_m', 'z_m'] + [f'rotation_{r}{c}' for r in range(3) for c in range(3)] + ['finger_open_m'],
        action_frame='MuJoCo world', phase_names=PHASE_NAMES,
        observation_alignment='obs[i], actions[i], obs[i+1]',
        training_selection='successful episode AND training_mask; return and close phases only'))
    successes, attempts, transitions = 0, 0, 0
    started = time.monotonic()
    status = 'running'
    summary = {}

    def save_summary():
        summary.update(status=status, requested_episodes=args.episodes,
                       successful_episodes=successes, attempts=attempts,
                       training_transitions=transitions, elapsed_s=time.monotonic()-started)
        write_json(args.output / 'summary.json', summary)

    pool = None
    try:
        if args.workers > 1:
            pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'))
        with (args.output / 'attempts.jsonl').open('w', buffering=1) as manifest:
            while successes < args.episodes and attempts < args.max_attempts:
                # Never launch more successes than requested, and keep only one
                # episode per worker in memory rather than accumulating a dataset.
                count = min(args.workers, args.episodes-successes, args.max_attempts-attempts)
                requests = [(args.seed+i, str(args.output), asdict(config), backend_config, args.save_failures)
                            for i in range(attempts, attempts+count)]
                results = pool.map(record_attempt, requests) if pool else map(record_attempt, requests)
                for result in results:
                    attempts += 1
                    successes += int(result['success'])
                    transitions += result['training_steps']
                    manifest.write(json.dumps(result, default=json_default) + '\n')
                    print(json.dumps(dict(seed=result['seed'], success=result['success'],
                                          failure=result['failure'], accepted=successes,
                                          requested=args.episodes)), flush=True)
                    save_summary()
        status = 'complete' if successes == args.episodes else 'attempt_limit'
    except BaseException:
        status = 'interrupted_or_error'
        raise
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        save_summary()
    print(json.dumps(summary, indent=2))
    return 0 if successes == args.episodes else 1


if __name__ == '__main__':
    raise SystemExit(main())
