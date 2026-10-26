#!/usr/bin/env python3
"""Structure-locked generative restyling of rendered frames, toward sim-to-real.

Stable Diffusion 1.5 img2img with depth and edge ControlNets (made few-step with
LCM-LoRA) repaints the simulator's shoulder and wrist renders to look like real
camera images. Geometry is held by the simulator itself: the depth control is
the exact rendered depth and the edge control is the outline of every object
from the simulator's segmentation -- shapes, not the sim's textures -- so the
table, arm and cubes keep their shape and place while surfaces are repainted. Cube and
gripper-finger pixels are then pasted back exactly from the render using the
simulator's segmentation, so the cubes' positions and colours -- which the
stacking order depends on -- and the jaws used to judge alignment are untouched.
Each episode uses one prompt and one noise seed for every frame, which limits
frame-to-frame flicker.

Two measurements check every restyle: outline recall (the share of simulator
object outlines with an image edge within 2 pixels, next to the same figure for
the plain render) and flicker (mean change between consecutive generated frames
relative to the sim's own change).

Preview a few frames before committing GPU time:

    python tools/sim2real_augment.py preview --episode data/scripted_stack/episode_0000 \\
        --frames 8 --output outputs/sim2real_preview.jpg

``export_scripted_dataset.py --restyle-fraction`` applies it during export.

Compare the published setting with stronger ones, including one styled after
real camera frames through an IP-Adapter (frames taken from the real dataset):

    python tools/sim2real_augment.py compare --episode data/scripted_stack/episode_0000 \\
        --output outputs/sim2real_compare.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CUBES = ('cube_red', 'cube_green', 'cube_blue')
# The jaws are pasted back too: they are what grasp alignment is judged by, and
# a fast restyle painted the wrist view's finger copper beside the red cube.
KEPT_BODIES = ('L_finger', 'R_finger')
# Every prompt names the robot's real finish: without it, a wooden table bleeds into the arm.
ARM = 'a white and grey aluminium robot arm with a black metal parallel gripper'
PROMPTS = (
    f'a photo of {ARM} on a light wooden table in a research lab, overhead fluorescent lighting',
    f'a photo of {ARM} over a desk with a laminate top, daylight from a window',
    f'a photo of {ARM} above a grey workbench in a workshop, soft shadows',
    f'a photo of {ARM} on a dark wooden table, warm indoor lighting, slight camera noise',
    f'a photo of {ARM} on a white table in a bright office, realistic, photographed',
    f'a photo of {ARM} on a plywood table in a garage, mixed lighting',
)
NEGATIVE = ('cartoon, 3d render, cgi, illustration, painting, blurry, distorted, text, watermark, '
            'wooden robot, bronze robot, copper robot, orange robot, '
            'extra cubes, extra blocks, coloured objects, green object, red object, blue object')


@dataclass(frozen=True)
class Settings:
    base: str = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    depth_controlnet: str = 'lllyasviel/control_v11f1p_sd15_depth'
    canny_controlnet: str = 'lllyasviel/control_v11p_sd15_canny'
    lcm_lora: str = 'latent-consistency/lcm-lora-sdv1-5'
    size: int = 512                 # SD 1.5's native resolution; output is resized back
    steps: int = 6
    strength: float = .75           # how much of the render is replaced
    guidance: float = 1.5
    depth_scale: float = .9
    canny_scale: float = .5
    # Fixed per-camera inverse-depth ranges (m), so brightness does not pulse.
    depth_range: tuple = (('shoulder', .6, 2.2), ('wrist', .03, 1.), ('overhead', .8, 3.))
    batch: int = 8
    # Background pixels (sky, floor, table) whose render did not change since
    # the previous frame keep the previous restyle, so a fixed camera's
    # background cannot shimmer. The robot is always regenerated whole: carrying
    # over only its still parts stitches one arm together from several frames.
    carry_static: bool = True
    static_threshold: int = 6       # max per-channel render change counted as still
    static_margin: int = 5          # px grown around anything that moved
    prompts: tuple = field(default=PROMPTS)
    # Optional style reference: a real camera frame steering the look through an
    # IP-Adapter. Ignored when the pipeline was built without one.
    ip_scale: float = 0.
    negative: str = NEGATIVE


REAL_DATASET = 'FoxNerdSaysMoo/panthera-gear-carrier-pin-real-20hz'
IP_ADAPTER = ('h94/IP-Adapter', 'models', 'ip-adapter_sd15.bin')
# Named variants for `compare`; 'published' is what the restyled dataset used.
VARIANTS = {
    'published': {},
    'bold': dict(strength=.92, steps=8, depth_scale=.7, canny_scale=.35),
    'real-photo style': dict(strength=.9, steps=8, depth_scale=.75, canny_scale=.4, ip_scale=.7),
}


def depth_image(depth: np.ndarray, settings: Settings, camera: str) -> np.ndarray:
    """Inverse depth on a fixed per-camera range, near = bright, as the depth ControlNet expects."""
    near, far = next((n, f) for name, n, f in settings.depth_range if name == camera)
    inverse = 1 / np.clip(depth, near, far)
    lo, hi = 1 / far, 1 / near
    gray = ((inverse - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def boundary_image(segment: np.ndarray) -> np.ndarray:
    """Object outlines from simulator segmentation: shapes without the sim's textures."""
    ids = segment.astype(np.int32)
    edges = np.zeros(ids.shape, bool)
    edges[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edges[1:, :] |= ids[1:, :] != ids[:-1, :]
    return np.repeat((edges * 255).astype(np.uint8)[..., None], 3, axis=2)


def boundary_recall(segment: np.ndarray, generated: np.ndarray, tolerance: int = 2) -> float:
    """Share of object outlines that still have an image edge within `tolerance` pixels.

    Repainted textures are expected to change; object outlines should not move.
    """
    outline = boundary_image(segment)[..., 0] > 0
    edges = cv2.Canny(cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY), 50, 120)
    near = cv2.dilate(edges, np.ones((2 * tolerance + 1,) * 2, np.uint8)) > 0
    return float((outline & near).sum() / max(outline.sum(), 1))


def flicker(generated: np.ndarray, reference: np.ndarray) -> float:
    """Consecutive-frame change of the restyle relative to the render's own change."""
    if len(generated) < 2:
        return float('nan')
    change = lambda frames: np.abs(np.diff(frames.astype(np.float32), axis=0)).mean()
    return float(change(generated) / max(change(reference), 1e-3))


class Restyler:
    def __init__(self, settings: Settings = Settings(), device: str | None = None, ip_adapter: bool = False):
        import torch
        from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetImg2ImgPipeline

        self.settings = settings
        self.device = device or ('cuda' if torch.cuda.is_available()
                                 else 'mps' if torch.backends.mps.is_available() else 'cpu')
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        controlnets = [ControlNetModel.from_pretrained(settings.depth_controlnet, torch_dtype=dtype),
                       ControlNetModel.from_pretrained(settings.canny_controlnet, torch_dtype=dtype)]
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            settings.base, controlnet=controlnets, torch_dtype=dtype, safety_checker=None,
            requires_safety_checker=False)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(settings.lcm_lora)
        pipe.fuse_lora()
        pipe.set_progress_bar_config(disable=True)
        if ip_adapter:
            repo, subfolder, weights = IP_ADAPTER
            pipe.load_ip_adapter(repo, subfolder=subfolder, weight_name=weights, torch_dtype=dtype)
        self.ip_adapter = ip_adapter
        self.pipe = pipe.to(self.device)
        self.torch = torch

    def episode(self, rgb: np.ndarray, depth: np.ndarray, segment: np.ndarray, keep: np.ndarray,
                seed: int, camera: str, background: np.ndarray | None = None, reference=None,
                controls: tuple | None = None) -> np.ndarray:
        """Restyle one camera's frames of one episode; `keep` marks pixels copied back.

        `controls` = (depth images, edge images), uint8 HxWx3 per frame, replaces the
        simulator's depth and segmentation outlines (for real footage).
        """
        from PIL import Image

        s, torch = self.settings, self.torch
        prompt = s.prompts[seed % len(s.prompts)]
        size = (s.size, s.size)
        height, width = rgb.shape[1:3]
        out = np.empty_like(rgb)
        style = {}
        if self.ip_adapter:
            # A loaded IP-Adapter always needs an image; scale 0 switches it off.
            self.pipe.set_ip_adapter_scale(s.ip_scale)
            blank = np.zeros((224, 224, 3), np.uint8)
            style['ip_adapter_image'] = Image.fromarray(blank if reference is None else reference)
        grow = np.ones((2 * s.static_margin + 1,) * 2, np.uint8)
        for start in range(0, len(rgb), s.batch):
            chunk = slice(start, start + s.batch)
            frames = rgb[chunk]
            count = len(frames)
            init = [Image.fromarray(cv2.resize(f, size, interpolation=cv2.INTER_CUBIC)) for f in frames]
            # Multi-ControlNet takes one batched tensor per conditioning, not nested lists.
            as_batch = lambda images: torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255
            if controls is not None:
                depth_control = as_batch([cv2.resize(d, size) for d in controls[0][chunk]])
                edge_control = as_batch([cv2.resize(e, size, interpolation=cv2.INTER_NEAREST)
                                         for e in controls[1][chunk]])
            else:
                depth_control = as_batch([cv2.resize(depth_image(d, s, camera), size) for d in depth[chunk]])
                edge_control = as_batch([cv2.resize(boundary_image(g), size, interpolation=cv2.INTER_NEAREST)
                                         for g in segment[chunk]])
            # The same noise for every frame of the episode limits flicker.
            generators = [torch.Generator('cpu').manual_seed(seed) for _ in range(count)]
            images = self.pipe(prompt=[prompt] * count, negative_prompt=[s.negative] * count,
                               image=init, control_image=[depth_control, edge_control],
                               strength=s.strength, num_inference_steps=s.steps,
                               guidance_scale=s.guidance,
                               controlnet_conditioning_scale=[s.depth_scale, s.canny_scale],
                               generator=generators, **style).images
            for i, image in enumerate(images):
                generated = cv2.resize(np.asarray(image), (width, height), interpolation=cv2.INTER_AREA)
                alpha = cv2.GaussianBlur(keep[start + i].astype(np.float32), (3, 3), 0)[..., None]
                frame = (alpha * frames[i] + (1 - alpha) * generated).astype(np.uint8)
                index = start + i
                if s.carry_static and index > 0:
                    moved = np.abs(rgb[index].astype(np.int16) - rgb[index - 1]).max(axis=2) > s.static_threshold
                    still = cv2.dilate(moved.astype(np.uint8), grow) == 0
                    if background is not None:
                        still &= background[index]
                    frame[still] = out[index - 1][still]
                out[index] = frame
        return out


def background_geom_ids(model) -> list[int]:
    """Scenery that never moves: the floor and every geom of the table body."""
    import mujoco
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'table')
    return [i for i in range(model.ngeom)
            if model.geom_bodyid[i] == table or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) == 'floor']


def scene_maps(sim, renderer, camera, kept_geoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth (m), per-pixel object ids (-1 for none) and the copied-back mask for the current state."""
    renderer.update_scene(sim.data, camera)
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    import mujoco
    is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    ids = np.where(is_geom, segmentation[..., 0], -1)
    return depth, ids, is_geom & np.isin(segmentation[..., 0], kept_geoms)


def set_frame(sim, arrays: dict, i: int) -> None:
    """The renderer's per-frame state assignment (teleop/render_vla_dataset.py)."""
    import mujoco
    sim.data.qpos[:] = sim.model.qpos0
    sim.data.qvel[:] = 0.
    sim.data.qpos[sim.arm_qadr] = arrays['q'][i]
    sim.data.qpos[sim.finger_qadr] = arrays['finger_q'][i]
    sim.data.ctrl[:] = arrays['ctrl'][i]
    sim.set_object_poses(arrays['obj_pos'][i], arrays['obj_quat'][i])
    mujoco.mj_forward(sim.model, sim.data)


def kept_geom_ids(model) -> list[int]:
    """Geoms copied back unchanged from the render: the cubes and the gripper fingers."""
    import mujoco
    cubes = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in CUBES]
    bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in KEPT_BODIES}
    fingers = [i for i in range(model.ngeom) if model.geom_bodyid[i] in bodies]
    return [i for i in cubes if i >= 0] + fingers


def restyle_rendered(dest: Path, sim, renderers, cameras, restyler: Restyler, seed: int) -> dict:
    """Restyle a rendered episode directory's JPEGs in place; returns its measurements."""
    with np.load(dest / 'trajectory.npz') as data:
        arrays = {key: data[key] for key in ('q', 'finger_q', 'ctrl', 'obj_pos', 'obj_quat')}
    cubes = kept_geom_ids(sim.model)
    report = dict(seed=seed, prompt=restyler.settings.prompts[seed % len(restyler.settings.prompts)])
    for name, renderer, camera in zip(('shoulder', 'wrist'), renderers, cameras):
        paths = sorted((dest / name).glob('*.jpg'))
        rgb = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths])
        depth, segment, keep = [], [], []
        for i in range(len(paths)):
            set_frame(sim, arrays, i)
            d, g, k = scene_maps(sim, renderer, camera, cubes)
            depth.append(d)
            segment.append(g)
            keep.append(k)
        segment = np.stack(segment)
        background = (segment < 0) | np.isin(segment, background_geom_ids(sim.model))
        restyled = restyler.episode(rgb, np.stack(depth), segment, np.stack(keep), seed, name, background)
        for path, frame in zip(paths, restyled):
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        sample = np.linspace(0, len(paths) - 1, min(20, len(paths))).astype(int)
        report[name] = dict(
            outline_recall=float(np.mean([boundary_recall(segment[i], restyled[i]) for i in sample])),
            render_outline_recall=float(np.mean([boundary_recall(segment[i], rgb[i]) for i in sample])),
            flicker=flicker(restyled, rgb))
    report['settings'] = {k: v for k, v in asdict(restyler.settings).items() if k != 'prompts'}
    (dest / 'restyle.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def preview(args):
    """Render a few frames of a native episode and restyle them into a comparison grid."""
    import os
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    from sim.panthera_env import PantheraSim
    from teleop.render_vla_dataset import (interpolate, recording_times, reconstruct_fingers,
                                           shoulder_camera, wrist_camera)

    meta = json.loads((args.episode / 'meta.json').read_text())
    sim = PantheraSim(ROOT / meta.get('scene', 'sim/panthera/scene.xml'))
    with np.load(args.episode / 'data.npz') as source:
        t, _ = recording_times(source, sim.dt)
        grid = np.linspace(t[0], t[-1], args.frames) if args.spread else \
            np.arange(args.frames) / 30 + t[len(t) // 3]
        arrays = {k: interpolate(t, source[k], grid, k.endswith('quat'))
                  for k in ('q', 'ctrl', 'obj_pos', 'obj_quat')}
        fingers, _ = reconstruct_fingers(source, t, sim)
        arrays['finger_q'] = interpolate(t, fingers, grid)
    renderers = [mujoco.Renderer(sim.model, 256, 256) for _ in range(2)]
    cameras = [shoulder_camera(sim.model), wrist_camera(sim.model)]
    cubes = kept_geom_ids(sim.model)
    restyler = Restyler(Settings(strength=args.strength, steps=args.steps, batch=args.batch,
                                 size=args.size, guidance=args.guidance))
    rows, report = [], {}
    for name, renderer, camera in zip(('shoulder', 'wrist'), renderers, cameras):
        rgb, depth, segment, keep = [], [], [], []
        for i in range(args.frames):
            set_frame(sim, arrays, i)
            renderer.update_scene(sim.data, camera)
            rgb.append(renderer.render().copy())
            d, g, k = scene_maps(sim, renderer, camera, cubes)
            depth.append(d)
            segment.append(g)
            keep.append(k)
        rgb, depth, segment, keep = np.stack(rgb), np.stack(depth), np.stack(segment), np.stack(keep)
        background = (segment < 0) | np.isin(segment, background_geom_ids(sim.model))
        clock = time.time()
        restyled = restyler.episode(rgb, depth, segment, keep, args.seed, name, background)
        report[name + '_seconds_per_image'] = round((time.time() - clock) / len(rgb), 3)
        report[name] = dict(
            outline_recall=float(np.mean([boundary_recall(g, b) for g, b in zip(segment, restyled)])),
            render_outline_recall=float(np.mean([boundary_recall(g, a) for g, a in zip(segment, rgb)])),
            flicker=flicker(restyled, rgb))
        for i in range(args.frames):
            rows.append(np.concatenate([rgb[i], depth_image(depth[i], restyler.settings, name),
                                        boundary_image(segment[i]), restyled[i]], axis=1))
    grid_image = np.concatenate(rows, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
    print(json.dumps(dict(prompt=restyler.settings.prompts[args.seed % len(PROMPTS)], **report), indent=2))


def real_references(episode: int = 0, cache: Path = ROOT / 'outputs/real_style') -> dict:
    """One mid-episode real overhead and wrist frame from the real dataset (~80 MB download)."""
    paths = {'shoulder': cache / f'episode{episode}_overhead.jpg', 'wrist': cache / f'episode{episode}_wrist.jpg'}
    if not all(p.is_file() for p in paths.values()):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        meta = pq.read_table(hf_hub_download(REAL_DATASET, 'meta/episodes/chunk-000/file-000.parquet',
                                             repo_type='dataset')).to_pylist()[episode]
        data = hf_hub_download(REAL_DATASET, f"data/chunk-{meta['data/chunk_index']:03d}/"
                               f"file-{meta['data/file_index']:03d}.parquet", repo_type='dataset')
        table = pq.read_table(data, columns=['episode_index', 'observation.images.overhead',
                                             'observation.images.wrist_port2'])
        rows = np.flatnonzero(table['episode_index'].to_numpy() == episode)
        row = int(rows[len(rows) // 2])
        cache.mkdir(parents=True, exist_ok=True)
        for name, column in (('shoulder', 'observation.images.overhead'), ('wrist', 'observation.images.wrist_port2')):
            paths[name].write_bytes(table[column][row].as_py()['bytes'])
    return {name: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for name, p in paths.items()}


def compare(args):
    """Restyle the same frames with each variant: render | published | bold | real-photo style."""
    import dataclasses
    import os
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    from sim.panthera_env import PantheraSim
    from teleop.render_vla_dataset import (interpolate, recording_times, reconstruct_fingers,
                                           shoulder_camera, wrist_camera)

    meta = json.loads((args.episode / 'meta.json').read_text())
    sim = PantheraSim(ROOT / meta.get('scene', 'sim/panthera/scene.xml'))
    with np.load(args.episode / 'data.npz') as source:
        t, _ = recording_times(source, sim.dt)
        grid = np.linspace(t[0], t[-1], args.frames + 2)[1:-1]
        arrays = {k: interpolate(t, source[k], grid, k.endswith('quat'))
                  for k in ('q', 'ctrl', 'obj_pos', 'obj_quat')}
        fingers, _ = reconstruct_fingers(source, t, sim)
        arrays['finger_q'] = interpolate(t, fingers, grid)
    references = real_references(args.real_episode)
    base = Settings(batch=args.batch)
    restyler = Restyler(base, ip_adapter=True)
    renderers = [mujoco.Renderer(sim.model, 256, 256) for _ in range(2)]
    cameras = [shoulder_camera(sim.model), wrist_camera(sim.model)]
    kept = kept_geom_ids(sim.model)
    sections, report = [], {}

    def label(image, text):
        image = image.copy()
        cv2.rectangle(image, (0, 0), (image.shape[1], 16), (0, 0, 0), -1)
        cv2.putText(image, text, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    for name, renderer, camera in zip(('shoulder', 'wrist'), renderers, cameras):
        rgb, depth, segment, keep = [], [], [], []
        for i in range(args.frames):
            set_frame(sim, arrays, i)
            renderer.update_scene(sim.data, camera)
            rgb.append(renderer.render().copy())
            d, g, k = scene_maps(sim, renderer, camera, kept)
            depth.append(d), segment.append(g), keep.append(k)
        rgb, depth, segment, keep = map(np.stack, (rgb, depth, segment, keep))
        background = (segment < 0) | np.isin(segment, background_geom_ids(sim.model))
        reference = cv2.resize(references[name], rgb.shape[2:0:-1])
        columns = [('sim render', rgb)]
        report[f'{name} sim render'] = dict(outline_recall=round(float(np.mean(
            [boundary_recall(g, a) for g, a in zip(segment, rgb)])), 3))
        for variant, overrides in VARIANTS.items():
            restyler.settings = dataclasses.replace(base, **overrides)
            clock = time.time()
            restyled = restyler.episode(rgb, depth, segment, keep, args.seed, name, background, reference)
            report[f'{name} {variant}'] = dict(
                outline_recall=round(float(np.mean([boundary_recall(g, b) for g, b in zip(segment, restyled)])), 3),
                seconds_per_image=round((time.time() - clock) / len(rgb), 2))
            columns.append((variant, restyled))
        columns.append(('real reference', np.repeat(reference[None], len(rgb), 0)))
        for i in range(args.frames):
            sections.append(np.concatenate([label(frames[i], f'{name}: {title}') if i == 0 else frames[i]
                                            for title, frames in columns], axis=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(np.concatenate(sections, 0), cv2.COLOR_RGB2BGR))
    print(json.dumps(report, indent=2))
    print(f'wrote {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('compare', help='same frames restyled with each variant, beside a real frame')
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--frames', type=int, default=3, help='frames spread across the episode')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch', type=int, default=Settings.batch, help='frames per diffusion call; 1 on a laptop')
    p.add_argument('--real-episode', type=int, default=0, help='real episode the style frames come from')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/sim2real_compare.jpg')
    p = sub.add_parser('preview', help='restyle a few frames of one native episode into a grid')
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--frames', type=int, default=6)
    p.add_argument('--spread', action='store_true', help='sample across the episode instead of consecutively')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--strength', type=float, default=Settings.strength)
    p.add_argument('--steps', type=int, default=Settings.steps)
    p.add_argument('--batch', type=int, default=Settings.batch, help='frames per diffusion call; 1 on a laptop')
    p.add_argument('--size', type=int, default=Settings.size)
    p.add_argument('--guidance', type=float, default=Settings.guidance)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/sim2real_preview.jpg')
    args = parser.parse_args()
    {'compare': compare, 'preview': preview}[args.command](args)


if __name__ == '__main__':
    main()
