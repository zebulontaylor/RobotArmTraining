"""Collect physically verified pin regrasp or held-pin alignment demos in MuJoCo."""
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
    seed, output, config, backend_config, save_failures, *task_option = request
    task = task_option[0] if task_option else 'regrasp'
    from sim.pin_regrasp_backend import MujocoPinBackend
    started = time.monotonic()
    if task in ('alignment', 'insertion'):
        from sim.pin_regrasp_backend import MujocoPinInsertionBackend
        from collection.pin_insertion import InsertionCollectionConfig, collect_insertion_episode
        backend = MujocoPinInsertionBackend(**backend_config, max_steps=config['max_steps'])
        collector, settings = collect_insertion_episode, InsertionCollectionConfig(**config)
    else:
        backend = MujocoPinBackend(**backend_config)
        collector, settings = collect_episode, CollectionConfig(**config)
    try:
        arrays, result = collector(backend, seed, settings)
        result['task'] = 'alignment' if task in ('alignment', 'insertion') else task
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
    parser.add_argument('--task', choices=('regrasp', 'alignment', 'insertion'), default='regrasp')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-attempts', type=int, default=500)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hz', type=int, help='Default: 30 for regrasp; alignment requires 20')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--no-images', action='store_true', help='State-only diagnostic collection')
    parser.add_argument('--save-failures', action='store_true')
    parser.add_argument('--config', type=Path, help='JSON overrides for the selected task collection settings')
    args = parser.parse_args(argv)
    if args.task == 'insertion':
        args.task = 'alignment'
    args.hz = args.hz if args.hz is not None else (20 if args.task == 'alignment' else 30)
    if args.task == 'alignment' and args.hz != 20:
        parser.error('Alignment runs at 20 Hz')
    if min(args.episodes, args.max_attempts, args.workers, args.hz, args.image_size) < 1 or args.seed < 0:
        parser.error('Counts, frequency and image size must be positive; seed must be nonnegative')
    if args.max_attempts < args.episodes:
        parser.error('--max-attempts must be at least --episodes')
    if args.hz > 1000:
        parser.error('--hz cannot exceed the 1000 Hz physics timestep')
    try:
        from collection.pin_insertion import InsertionCollectionConfig
        config_class = InsertionCollectionConfig if args.task == 'alignment' else CollectionConfig
        config = config_class(**(json.loads(args.config.read_text()) if args.config else {}))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.output.exists() and not args.output.is_dir():
        parser.error('Output path must be a directory')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty; choose a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    backend_config = dict(hz=args.hz, image_size=args.image_size, images=not args.no_images)
    from collection.pin_insertion import PHASE_NAMES as INSERTION_PHASES
    insertion = args.task == 'alignment'
    if insertion:
        backend_config['scene_xy_m'] = config.scene_xy_m
    write_json(args.output / 'config.json', dict(
        schema_version=3 if insertion else 2, task=args.task,
        backend='MujocoPinAlignmentBackend' if insertion else 'MujocoPinBackend', backend_config=backend_config,
        setup_profile='photo_20261027', pose_profile='sim/actuator/pin_reset_profile.json' if insertion else 'sim/actuator/pin_regrasp_profile.json',
        collection=asdict(config), seed=args.seed, requested_episodes=args.episodes,
        max_attempts=args.max_attempts, workers=args.workers,
        action_fields=([f'{side}_{field}' for side in ('left', 'right')
                        for field in ('dx', 'dy', 'dz', 'rx', 'ry', 'rz', 'grip_delta')] if insertion else ['x_m', 'y_m', 'z_m'] + [f'rotation_{r}{c}' for r in range(3) for c in range(3)] + ['finger_open_m']),
        action_frame='MuJoCo world', phase_names=INSERTION_PHASES if insertion else PHASE_NAMES,
        action_contract=(dict(type='normalized_increment', clip=[-1, 1],
                             left_translation_m=.001, right_translation_m=.00025,
                             left_rotation_deg=.5, right_rotation_deg=.1, gripper_fraction=.05)
                         if insertion else dict(type='absolute_tool_pose')),
        observation_alignment='obs[i], actions[i], obs[i+1]',
        alignment_mode='alignment_only' if insertion else None,
        training_selection=('successful run AND training_mask; return phases only' if insertion
                            else 'successful episode AND training_mask; return and close phases only')))
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
                requests = [(args.seed+i, str(args.output), asdict(config), backend_config, args.save_failures, args.task)
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
