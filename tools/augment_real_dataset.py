#!/usr/bin/env python3
"""Augment the real two-arm demonstrations by restyling their table and background.

The motions are real and cannot be edited (the frames show the actual poses),
so this only changes how the scene looks. Per frame:

- a monocular depth model (Depth Anything V2) estimates depth;
- the table is modelled as a smooth surface fitted to the depth of wood-coloured
  pixels; everything that rises above it (arms, grippers, fingers, carrier, gear,
  pin, operator hands), plus anything not wood-coloured that is not farther than
  the table, is KEPT and copied back exactly; glare and small flat patches are not;
- leftover wood beside the kept objects (shadowed, so the table test misses it, and
  read as raised because monocular depth bleeds near tall objects) is repainted too,
  on half the copies by default (--wood-cleanup); the copy-back edge is blended
  with a guided filter;
- the rest (table and room) is repainted by the structure-locked diffusion
  restyle of tools/sim2real_augment.py, guided by that depth and the image's own
  edges, with prompts for other tables, rooms and lighting.

Output is a LeRobot dataset in the real dataset's exact format with every
original episode plus restyled copies (by default two per demo, one realistic and
one wild look); states, actions, velocities and efforts are copied unchanged.
Rendering resumes like the sim export. Restyled copies keep their depth and masks
(maps.npz) so `relight` can add lighting and shadow changes later without diffusion.

    python tools/augment_real_dataset.py preview --output outputs/real_aug_preview.jpg
    # every demo + a realistic and a wild restyled copy of each, no lighting change yet
    python tools/augment_real_dataset.py export --output /workspace/real_aug/lerobot \\
        --work /workspace/real_aug/rendered --copies 2 --workers 4
    # then lighting and shadows without diffusion (CPU, uses the stored depth and masks)
    python tools/augment_real_dataset.py relight --source /workspace/real_aug/lerobot \\
        --output /workspace/real_aug/lerobot_relit --work /workspace/real_aug/relit
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

REAL = 'FoxNerdSaysMoo/panthera-gear-carrier-pin-real-20hz'
CAMERAS = ('overhead', 'wrist_port2', 'wrist_port3')
NUMERIC = ('observation.state', 'action', 'observation.velocity', 'observation.effort')
DEPTH_MODEL = 'depth-anything/Depth-Anything-V2-Small-hf'
SCENES = ('a light maple butcher-block table in a university lab', 'a dark walnut wooden desk in a home office',
          'a white laminate workbench in a bright clean room', 'a grey steel workbench in a machine shop',
          'a black rubber mat on a workshop table', 'a pale plywood table in a cluttered garage',
          'a green cutting mat on a lab bench', 'a polished concrete counter in a warehouse',
          'a blue anti-static mat on an electronics bench', 'a white marble countertop in a kitchen',
          'a red-painted metal table in a factory', 'a cork board surface in a classroom',
          'a glossy black desk in a dark office', 'a stainless steel table in a clean lab',
          'a scratched old wooden workbench in a basement', 'a beige plastic folding table in a hallway',
          'a dark grey felt mat on an office desk', 'a yellow-orange workbench in a makerspace')
LIGHTS = ('bright fluorescent ceiling light', 'warm tungsten lamp light, soft shadows',
          'cool overcast daylight from a window', 'harsh afternoon sunlight, long shadows',
          'dim evening light, a single desk lamp', 'even studio lighting', 'blue-tinted LED panel light',
          'golden hour sunlight through blinds, striped shadows', 'low light, mostly in shadow',
          'strong side light from a window, high contrast')
ROBOT = ('two white and grey aluminium robot arms with white 3D printed fin-ray gripper fingers and black brackets, '
         'assembling a small black gear and a steel pin into a silver machined carrier')
NEGATIVE = ('cartoon, 3d render, cgi, illustration, painting, text, watermark, extra robot arms, extra hands, '
            'distorted hands, extra gears')


# Most restyled episodes stay plausible (muted, everyday surfaces); some are wild, which
# teaches the policy that the background does not matter.
REALISTIC_SHARE = .5
REALISTIC_PALETTES = ('grey and white', 'light wood', 'dark wood', 'charcoal black', 'beige and cream',
                      'cool grey', 'brown and tan', 'off-white', 'sage green', 'navy blue', 'terracotta',
                      'slate blue', 'olive green', 'burgundy', 'muted teal', 'warm brown', 'dusty blue',
                      'mustard and grey')
WILD_PALETTES = ('teal and white', 'deep blue', 'forest green', 'warm red', 'pastel pink', 'navy and orange',
                 'purple', 'mustard yellow', 'neon blue')


def prompts(realistic: bool = True):
    palettes = REALISTIC_PALETTES if realistic else WILD_PALETTES
    return tuple(f'a webcam photo of {ROBOT} on {scene}, {palette} colour scheme, {light}'
                 for scene in SCENES for light in LIGHTS for palette in palettes)


def is_realistic(seed: int) -> bool:
    return bool(np.random.default_rng([seed, 7]).random() < REALISTIC_SHARE)


def scramble_background(frames_rgb, keep, seed: int, realistic: bool | None = None):
    """Replace the table and room with a random colour field or texture (one per episode),
    so the restyle starts from a new palette instead of the original wood."""
    realistic = is_realistic(seed) if realistic is None else realistic
    rng = np.random.default_rng(seed)
    count, height, width = keep.shape
    y, x = np.mgrid[0:height, 0:width].astype(np.float32) / max(height, width)
    hsv = lambda h, s, v: cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)
    if realistic:
        # Everyday surfaces: neutral greys/whites/blacks, wood and desk tones, or muted
        # colours (painted desks, cutting mats, laminate).
        pick = rng.random()
        if pick < .35:
            first = hsv(rng.integers(0, 180), rng.integers(0, 25), rng.integers(30, 235))
        elif pick < .65:
            first = hsv(rng.integers(8, 22), rng.integers(40, 150), rng.integers(50, 220))
        else:
            first = hsv(rng.integers(0, 180), rng.integers(35, 120), rng.integers(50, 200))
        second = np.clip(first * rng.uniform(.75, 1.15), 0, 255)
        kind = str(rng.choice(['solid', 'gradient', 'planks', 'blotches']))
    else:
        first = hsv(rng.integers(0, 180), rng.integers(0, 230), rng.integers(25, 245))
        second = hsv(rng.integers(0, 180), rng.integers(0, 230), rng.integers(25, 245))
        kind = str(rng.choice(['solid', 'gradient', 'planks', 'tiles', 'blotches']))
    angle = rng.uniform(0, np.pi)
    u = x * np.cos(angle) + y * np.sin(angle)
    if kind == 'solid':
        weight = np.zeros_like(x)
    elif kind == 'gradient':
        weight = (u - u.min()) / max(np.ptp(u), 1e-6)
    elif kind == 'planks':
        weight = (np.floor(u * rng.uniform(6, 20)) % 2) * .6 + .2 * np.sin(u * rng.uniform(40, 120)) ** 2
    elif kind == 'tiles':
        n = rng.uniform(4, 12)
        weight = ((np.floor(x * n) + np.floor(y * n)) % 2) * .8
    else:
        noise = cv2.resize(rng.random((8, 8)).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
        weight = np.clip(noise, 0, 1)
    texture = first * (1 - weight[..., None]) + second * weight[..., None]
    texture += rng.normal(0, 6, texture.shape)
    texture = np.clip(texture, 0, 255).astype(np.uint8)
    out = frames_rgb.copy()
    out[~keep] = np.broadcast_to(texture, frames_rgb.shape)[~keep]
    return out, f"{'realistic' if realistic else 'wild'} {kind}"


# ---------------------------------------------------------------- masks -----

def wood(image_bgr):
    hsv = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 8) & (h <= 32) & (s >= 35) & (v >= 90)


def table_colour(image_bgr):
    """Table-coloured pixels: the wood hue, learned per frame so pale, washed-out wood still counts."""
    lab = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    hue, chroma = np.degrees(np.arctan2(b, a)), np.hypot(a, b)
    seed = wood(image_bgr)
    if seed.sum() < 200:
        return seed
    table_hue = np.median(hue[seed])
    return seed | ((np.abs((hue - table_hue + 180) % 360 - 180) < 22) & (chroma > 5) & (lab[..., 0] > 60))


def wood_any_light(image_bgr, spread=14.):
    """Pixels with the table's wood hue at any brightness, shadowed wood included (the
    table test above wants lit wood). Low-chroma pixels (white fingers, grey metal, black
    brackets) never count."""
    lab = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    hue, chroma = np.degrees(np.arctan2(b, a)), np.hypot(a, b)
    seed = wood(image_bgr)
    if seed.sum() < 200:
        return np.zeros(seed.shape, bool)
    table_hue = np.median(hue[seed])
    return (np.abs((hue - table_hue + 180) % 360 - 180) < spread) & (chroma > 6) & (lab[..., 0] > 20)


def table_surface(disparity, table, grid=64, sigma=4.):
    """Smooth disparity of the table, filled in under objects (follows fisheye and perspective)."""
    height, width = disparity.shape
    d = cv2.resize(disparity, (grid, grid), interpolation=cv2.INTER_AREA)
    w = cv2.resize(table.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    w = (w > .6).astype(np.float32)
    surface = d.copy()
    for s in (sigma, sigma * 2, sigma * 4):  # normalised convolution, widening to bridge large objects
        num = cv2.GaussianBlur(d * w, (0, 0), s)
        den = cv2.GaussianBlur(w, (0, 0), s)
        estimate = num / np.maximum(den, 1e-4)
        surface = np.where((w > 0) | (den > .05), estimate, surface) if s == sigma else \
            np.where(den > .02, np.where(w > 0, surface, estimate), surface)
    return cv2.resize(surface, (width, height), interpolation=cv2.INTER_LINEAR)


def keep_mask(image_bgr, disparity, margin=.06, details=False, clean=True):
    """Pixels to copy back unchanged: everything rising above the table, or not table-coloured and not farther.

    With `details`, also returns the height above the table surface and the table-coloured pixels.
    """
    lo, hi = np.percentile(disparity, [1, 99])
    disparity = (disparity - lo) / max(hi - lo, 1e-6)
    hsv = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    bright = cv2.dilate(((hsv[..., 2] > 235) & (hsv[..., 1] < 45)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    table = table_colour(image_bgr)
    if table.sum() < 200:  # no visible table: keep everything rather than guess
        keep = np.ones(disparity.shape, bool)
        return (keep, np.zeros_like(disparity), table) if details else keep
    surface = table_surface(disparity, table)
    residual = disparity - surface
    # Glare lies flat on the table; bright white things that rise from it (fingers) are objects.
    glare = bright & (residual < margin)
    # The room and walls sit farther than the table (lower disparity), so they are repainted.
    keep = ((residual > margin) | (~table & ~glare & (residual > -.03))).astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(keep)
    lightness = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    for i in range(1, count):  # small patches that do not rise from the table: grain or glare edges...
        part = labels == i
        if stats[i, 4] < 250 and np.median(residual[part]) < margin:
            ring = cv2.dilate(part.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~part
            # ...unless they stand out from the table around them, like the pin lying on it.
            if abs(lightness[part].mean() - lightness[ring].mean()) < 25:
                keep[part] = 0
    if clean:
        # Wood that the table test missed (in an arm's shadow, or the pale far edge) and
        # that does not rise from the table: drop islands made mostly of it, and strip it
        # from the rims of kept objects.
        # Monocular depth bleeds near tall objects, so wood beside an arm reads a few cm
        # high; only wood-coloured pixels well above the table (hands, forearms) stay.
        loose = (table | wood_any_light(image_bgr, spread=24.)) & (residual < 2 * margin)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(keep)
        for i in range(1, count):
            part = labels == i
            if stats[i, 4] < 4000 and loose[part].mean() > .6:
                keep[part] = 0
        rim = cv2.dilate((keep == 0).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        keep[loose & rim] = 0
        keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
    # Small parts (the pin, the loose gear) get a wider margin: a restyle bleeding a few
    # pixels into them changes what they look like.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(keep)
    small = np.isin(labels, [i for i in range(1, count) if stats[i, 4] < 600])
    keep = keep.astype(bool) | cv2.dilate(small.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return (keep, residual, table) if details else keep


def object_masks(frames_rgb, disparity, wanted=None, clean=True):
    """Kept-pixel masks (keep_mask) for the `wanted` frames (default all); `disparity` is indexed by frame."""
    wanted = range(len(frames_rgb)) if wanted is None else wanted
    return np.stack([keep_mask(cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR), disparity[i], clean=clean)
                     for i in wanted])


def guided_alpha(image_rgb, keep, radius=4, eps=1e-3):
    """Soft copy-back weight: the hard mask snapped to the image's own edges (a guided filter)."""
    guide = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    p = keep.astype(np.float32)
    box = lambda x: cv2.boxFilter(x, -1, (2 * radius + 1,) * 2)
    mean_i, mean_p = box(guide), box(p)
    a = (box(guide * p) - mean_i * mean_p) / (box(guide * guide) - mean_i ** 2 + eps)
    b = mean_p - a * mean_i
    alpha = np.clip(box(a) * guide + box(b), 0, 1)
    # Inside the mask stays exactly the original; only the edge band is blended.
    inside = cv2.erode(keep.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return np.where(inside, 1., alpha)


def soft_composite(original_rgb, restyled_rgb, keep):
    out = np.empty_like(restyled_rgb)
    for i, (o, r, k) in enumerate(zip(original_rgb, restyled_rgb, keep)):
        alpha = guided_alpha(o, k)[..., None]
        out[i] = np.clip(alpha * o + (1 - alpha) * r, 0, 255).astype(np.uint8)
    return out


class Depth:
    def __init__(self):
        import torch
        from transformers import pipeline
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.pipe = pipeline('depth-estimation', model=DEPTH_MODEL, device=device)

    def __call__(self, frames_rgb):
        from PIL import Image
        out = []
        for result, frame in zip(self.pipe([Image.fromarray(f) for f in frames_rgb], batch_size=16), frames_rgb):
            d = np.asarray(result['depth'], np.float32)
            out.append(cv2.resize(d, (frame.shape[1], frame.shape[0])))
        return np.stack(out)


def controls(frames_rgb, disparity, keep=None):
    """Depth control (near = bright, scaled per episode so it does not pulse) and image edges.

    With `keep`, edges are limited to the kept objects and their outlines, so the table
    grain and the room do not pin the repaint to the original surface.
    """
    lo, hi = np.percentile(disparity, [1, 99])
    depth = np.clip((disparity - lo) / max(hi - lo, 1e-6), 0, 1)
    depth_images = np.repeat((depth * 255).astype(np.uint8)[..., None], 3, -1)
    edges = []
    for i, f in enumerate(frames_rgb):
        e = cv2.Canny(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), 80, 160)
        if keep is not None:
            near = cv2.dilate(keep[i].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            e = np.where(near, e, 0).astype(np.uint8)
        edges.append(e)
    return depth_images, np.repeat(np.stack(edges)[..., None], 3, -1)


def sample_lighting(seed: int) -> dict:
    """One episode's whole-frame lighting change, robot included."""
    rng = np.random.default_rng(seed)
    cast = rng.choice(['neutral', 'warm', 'cool', 'green', 'blue', 'neutral'])
    tint = {'neutral': [1, 1, 1], 'warm': [1.18, 1.0, .78], 'cool': [.85, .97, 1.18],
            'green': [.9, 1.12, .9], 'blue': [.8, .92, 1.25]}[cast]
    return dict(gain=float(rng.uniform(.55, 1.35)), gamma=float(rng.uniform(.8, 1.3)),
                tint=(np.array(tint) * rng.uniform(.95, 1.05, 3)).tolist(), cast=str(cast),
                light_azimuth=float(rng.uniform(0, 360)), light_elevation=float(rng.uniform(15, 70)),
                shading=float(rng.uniform(.15, .55)),
                gradient=float(rng.uniform(0, .45)), gradient_angle=float(rng.uniform(0, 360)),
                vignette=float(rng.uniform(0, .45)), shadows=int(rng.integers(0, 3)),
                shadow_seed=int(rng.integers(2**31)), cast_shadow=float(rng.uniform(.3, .6)))


def relight(frames_rgb, disparity, look: dict, keep=None, camera='wrist'):
    """Apply one episode's lighting to every frame: directional shading from the estimated
    depth, shadows cast by the kept objects onto the table away from the light, a light
    falloff across the frame, colour cast, exposure and drifting soft shadows.

    Monocular depth comes with its own scale and offset in every frame, and the kept-object
    masks wobble a little, so each frame's depth is normalised on its own and the shading is
    smoothed over time: strongly for the fixed overhead camera (where any change reads as
    flicker), lightly for the moving wrist cameras; cast shadows are smoothed lightly so they
    keep up with the arms."""
    count, height, width = disparity.shape
    az, el = np.radians(look['light_azimuth']), np.radians(look['light_elevation'])
    light = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    ga = np.radians(look['gradient_angle'])
    ramp = ((x - width / 2) * np.cos(ga) + (y - height / 2) * np.sin(ga)) / (width / 2)
    gradient = 1 + look['gradient'] * np.clip(ramp, -1, 1)
    radius = np.hypot((x - width / 2) / (width / 2), (y - height / 2) / (height / 2)) / np.sqrt(2)
    vignette = 1 - look['vignette'] * radius ** 2
    rng = np.random.default_rng(look['shadow_seed'])
    blobs = []
    for _ in range(look['shadows']):
        keys = max(2, count // 60)
        points = rng.uniform(-.1, 1.1, (keys, 2)) * [width, height]
        path = np.stack([np.interp(np.linspace(0, keys - 1, count), np.arange(keys), points[:, k]) for k in range(2)], 1)
        blobs.append((path, rng.uniform([30, 12], [110, 45]), rng.uniform(0, 180), rng.uniform(.3, .6)))
    # Weight of the new frame in each running layer: the overhead camera is fixed, so its
    # depth shading can be averaged hard; cast shadows follow the arms, so they stay quick.
    shade_rate = .05 if camera == 'overhead' else .5
    shadow_rate = .5
    shade_layer = shadow_layer = None
    out = np.empty_like(frames_rgb)
    for i, (frame, d) in enumerate(zip(frames_rgb, disparity)):
        lo, hi = np.percentile(d, [1, 99])
        d = cv2.GaussianBlur(((d - lo) / max(hi - lo, 1e-6)).astype(np.float32), (0, 0), 5)
        gx, gy = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=5), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=5)
        normal = np.dstack([-gx * 4, -gy * 4, np.ones_like(d)])
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        lambert = np.clip(normal @ light, 0, 1)
        shade = 1 - look['shading'] + look['shading'] * 1.25 * lambert
        shadow = np.zeros((height, width), np.float32)
        for path, axes, angle, darkness in blobs:
            cv2.ellipse(shadow, tuple(int(v) for v in path[i]), tuple(int(a) for a in axes), angle, 0, 360, darkness, -1)
        if blobs:
            shadow = cv2.GaussianBlur(shadow, (0, 0), 18)
        if keep is not None and look.get('cast_shadow'):
            # The repaint removed the real shadows on the table: cast the arms, parts and hands
            # onto it, away from the new light, longer when the light is low.
            reach = min(60., 18. / max(np.tan(el), .25))
            shift = np.float32([[1, 0, -np.cos(az) * reach], [0, 1, np.sin(az) * reach]])
            caster = cv2.warpAffine(keep[i].astype(np.float32), shift, (width, height))
            cast = cv2.GaussianBlur(caster, (0, 0), 3 + reach / 8) * look['cast_shadow']
            shadow = np.maximum(shadow, np.where(keep[i], 0, cast))
        shadow = 1 - np.clip(shadow, 0, .7)
        shade_layer = shade if shade_layer is None else (1 - shade_rate) * shade_layer + shade_rate * shade
        shadow_layer = shadow if shadow_layer is None else (1 - shadow_rate) * shadow_layer + shadow_rate * shadow
        factor = (shade_layer * shadow_layer * gradient * vignette)[..., None] * look['gain']
        image = np.clip(frame.astype(np.float32) / 255 * factor * np.array(look['tint']), 0, 1) ** look['gamma']
        out[i] = (image * 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- data ------

def fetch(root: Path, revision: str | None):
    from huggingface_hub import HfApi, snapshot_download
    revision = revision or HfApi().dataset_info(REAL).sha
    snapshot_download(REAL, repo_type='dataset', revision=revision, local_dir=root,
                      allow_patterns=['data/**', 'meta/**'])
    return revision


def episode_rows(root: Path, episode: int):
    """Numeric arrays and JPEG bytes per camera for one real episode."""
    import pyarrow.parquet as pq
    import pandas as pd
    meta = pd.concat([pq.read_table(f).to_pandas() for f in sorted(root.glob('meta/episodes/*/*.parquet'))])
    row = meta[meta['episode_index'] == episode].iloc[0]
    path = root / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    table = pq.read_table(path).to_pandas()
    table = table[table['episode_index'] == episode].sort_values('frame_index')
    numeric = {k: np.stack(table[k].to_numpy()).astype(np.float32) for k in NUMERIC}
    images = {c: [cell['bytes'] for cell in table[f'observation.images.{c}']] for c in CAMERAS}
    return numeric, images


def real_features(root: Path) -> dict:
    """The real dataset's own feature schema, minus LeRobot's bookkeeping columns."""
    info = json.loads((root / 'meta/info.json').read_text())
    skip = {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
    return {k: {kk: (tuple(vv) if kk == 'shape' else vv) for kk, vv in v.items() if kk in ('dtype', 'shape', 'names')}
            for k, v in info['features'].items() if k not in skip}


def save_prepared_episode(dataset, image_stats):
    """Save an episode whose image stats were computed in the worker; batch the JPEG embedding."""
    from unittest.mock import patch
    import lerobot.datasets.lerobot_dataset as lerobot_io
    from datasets.table import embed_table_storage
    from lerobot.datasets.compute_stats import compute_episode_stats

    def prepared_stats(buffer, schema):
        numeric = {k: v for k, v in schema.items() if v['dtype'] != 'image'}
        return {**compute_episode_stats({k: buffer[k] for k in numeric}, numeric), **image_stats}

    def embed_batched(table):
        fmt = table.format
        return table.with_format('arrow').map(embed_table_storage, batched=True, batch_size=256).with_format(**fmt)

    with patch.object(lerobot_io, 'compute_episode_stats', prepared_stats), \
            patch.object(lerobot_io, 'embed_images', embed_batched):
        dataset.save_episode()


def decode(jpegs):
    return np.stack([cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB) for b in jpegs])


# ---------------------------------------------------------------- jobs ------

def restyle_settings(args):
    from tools.sim2real_augment import Settings
    # carry_static is for the sim's fixed cameras. Here the restyle input is the scrambled
    # background, a texture identical in every frame, so it would freeze the first frame's
    # repaint and paste stale pieces forward as the wrist cameras move (mosaic blocks).
    return Settings(batch=args.batch, steps=args.steps, strength=args.strength, guidance=args.guidance,
                    size=args.size, depth_scale=args.depth_scale, canny_scale=args.canny_scale, ip_scale=0.,
                    prompts=prompts(), negative=NEGATIVE, carry_static=False)


MAP_SIZE = 128  # stored depth resolution for the later lighting pass


def save_maps(path, maps):
    """Per camera: kept-pixel masks (bit-packed) and depth (uint8, MAP_SIZE), for `relight`."""
    out = {}
    for camera, (keep, disparity) in maps.items():
        lo, hi = np.percentile(disparity, [1, 99])
        d = np.clip((disparity - lo) / max(hi - lo, 1e-6), 0, 1)
        out[f'{camera}_disparity'] = np.stack([cv2.resize(x, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_AREA)
                                               for x in d.astype(np.float32)]) * 255
        out[f'{camera}_disparity'] = out[f'{camera}_disparity'].astype(np.uint8)
        out[f'{camera}_keep'] = np.packbits(keep, axis=-1)
        out[f'{camera}_shape'] = np.array(keep.shape)
    np.savez_compressed(path, **out)


def load_maps(path):
    with np.load(path) as z:
        maps = {}
        for camera in CAMERAS:
            count, height, width = z[f'{camera}_shape']
            keep = np.unpackbits(z[f'{camera}_keep'], axis=-1, count=width).astype(bool)
            disparity = np.stack([cv2.resize(d, (width, height), interpolation=cv2.INTER_LINEAR)
                                  for d in z[f'{camera}_disparity'].astype(np.float32) / 255])
            maps[camera] = (keep, disparity)
    return maps


def compute_maps(frames_by_camera, clean=True):
    """Depth and kept-pixel masks per camera (loads the depth model once per process).

    clean: repaint leftover wood beside the kept objects (see keep_mask).
    """
    global depth_model
    if 'depth_model' not in globals():
        depth_model = Depth()
    maps = {}
    for camera, frames in frames_by_camera.items():
        disparity = depth_model(frames)
        maps[camera] = (object_masks(frames, disparity, clean=clean), disparity)
    return maps


def write_folder(root, dest, numeric, frames_or_jpegs, report):
    """One output episode folder: JPEGs per camera, numeric arrays, image stats, done.json."""
    import pickle
    from lerobot.datasets.compute_stats import compute_episode_stats
    count = len(numeric['action'])
    for camera, frames in frames_or_jpegs.items():
        (dest / camera).mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            if isinstance(frame, bytes):
                (dest / camera / f'{i:05d}.jpg').write_bytes(frame)
            else:
                cv2.imwrite(str(dest / camera / f'{i:05d}.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
    image_features = {k: v for k, v in real_features(root).items() if v['dtype'] == 'image'}
    buffer = {key: [str(dest / key.rsplit('.', 1)[-1] / f'{j:05d}.jpg') for j in range(count)]
              for key in image_features}
    np.savez(dest / 'numeric.npz', **numeric)
    (dest / 'image_stats.pkl').write_bytes(pickle.dumps(compute_episode_stats(buffer, image_features)))
    (dest / 'done.json').write_text(json.dumps(report) + '\n')


CLAIM_STALE_S = 2 * 3600  # a claim older than this (a crashed or stopped pod) can be taken over


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


def claim(dest):
    """May this process render `dest`? Not if it is finished, or another process is rendering it.

    Several processes, on one machine or several sharing a network volume, can use one --work
    folder: each episode is claimed with an exclusive marker file holding `host:pid` before
    rendering. A claim is taken over only when abandoned: on this machine, when its process is
    no longer running; from another machine, when older than CLAIM_STALE_S. Taking over is
    atomic: the stale marker is renamed aside, which only one process can do, before a new one
    is created exclusively.
    """
    import socket
    if (dest / 'done.json').exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    mark = dest.parent / (dest.name + '.claim')
    host, pid = socket.gethostname(), os.getpid()

    def create():
        try:
            fd = os.open(mark, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, f'{host}:{pid}'.encode())
        os.close(fd)
        return True

    if create():
        return True
    try:
        seen = mark.read_text()
        owner_host, owner_pid = seen.rsplit(':', 1)
        owner_pid, age = int(owner_pid), time.time() - mark.stat().st_mtime
    except (OSError, ValueError):
        return False
    if owner_host == host and owner_pid == pid:
        return True  # already ours
    abandoned = (owner_host == host and not _alive(owner_pid)) or (owner_host != host and age > CLAIM_STALE_S)
    if not abandoned:
        return False
    aside = mark.with_name(f'{mark.name}.stale-{host}-{pid}')
    try:
        os.rename(mark, aside)
    except FileNotFoundError:
        return False  # another process took it over first
    if aside.read_text() != seen:  # it was replaced by a fresh claim in between: put it back
        os.rename(aside, mark)
        return False
    aside.unlink()
    return create()


def episode_ready(dest):
    """Finished, and still holding all its JPEGs (writing a dataset without --keep-jpegs removes them)."""
    try:
        frames = json.loads((dest / 'done.json').read_text())['frames']
    except (OSError, ValueError, KeyError):
        return False
    return all(len(list((dest / camera).glob('*.jpg'))) == frames for camera in CAMERAS)


def reopen_stripped(dests):
    """Finished folders whose JPEGs an earlier dataset write removed: mark them for rendering again."""
    stripped = [d for d in dests if (d / 'done.json').exists() and not episode_ready(d)]
    for d in stripped:
        (d / 'done.json').unlink()
        (d.parent / (d.name + '.claim')).unlink(missing_ok=True)
    return stripped


def check_settings(work, contract):
    """Record a run's settings in its --work folder, refusing a folder made with other settings
    (its finished episodes would otherwise be reused as if they matched)."""
    path = work / 'settings.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(contract, default=str)):
        raise SystemExit(f'{work} was made with other settings or another source; use a fresh --work folder')
    work.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, default=str) + '\n')


def output_matches(output, dests):
    """Is `output` already the complete dataset of exactly these episode folders?"""
    try:
        names = list(json.loads((output / 'meta/provenance.json').read_text())['episodes'])
    except (OSError, ValueError, KeyError):
        return False
    return (output / 'COMPLETE').exists() and names == [d.name for d in dests]


def render_job(root, dest, episode, restyle_seed, settings, args_relight=False, args_scramble=True,
               soft_edges=True, realistic=None, clean=True):
    """Write one output episode (original JPEG bytes, or restyled frames) with its image stats.

    Restyled episodes also keep their depth and masks (maps.npz) for the `relight` pass.
    """
    global restyler
    from dataclasses import replace
    if not claim(dest):
        return  # finished, or another pod is rendering it
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    numeric, images = episode_rows(root, episode)
    report = dict(source_episode=episode, frames=len(numeric['action']), restyled=restyle_seed is not None)
    if restyle_seed is None:
        write_folder(root, dest, numeric, images, report)
        return
    if 'restyler' not in globals():
        from tools.sim2real_augment import Restyler
        restyler = Restyler(settings)
    realistic = is_realistic(restyle_seed) if realistic is None else realistic
    frames = {camera: decode(images[camera]) for camera in CAMERAS}
    maps = compute_maps(frames, clean)
    look = sample_lighting(restyle_seed)  # same lighting for all three cameras of the episode
    out = {}
    for camera in CAMERAS:
        keep, disparity = maps[camera]
        source = frames[camera]
        if args_scramble:  # new palette for the table and room; kept pixels stay the originals
            source, report['scramble'] = scramble_background(frames[camera], keep, restyle_seed, realistic)
        restyler.settings = replace(settings, prompts=prompts(realistic))
        restyled = restyler.episode(source, None, None, keep, restyle_seed, camera, ~keep,
                                    controls=controls(frames[camera], disparity, keep))
        if soft_edges:
            restyled = soft_composite(frames[camera], restyled, keep)
        if args_relight:
            restyled = relight(restyled, disparity, look, keep, camera)
            report['lighting'] = look
        report[f'{camera}_kept_share'] = float(keep.mean())
        out[camera] = restyled
    save_maps(dest / 'maps.npz', maps)
    chosen = prompts(realistic)
    report.update(restyle_seed=restyle_seed, prompt=chosen[restyle_seed % len(chosen)], wood_cleanup=clean,
                  soft_edges=soft_edges, realistic=realistic,
                  restyle_settings={k: v for k, v in asdict(settings).items() if k != 'prompts'})
    write_folder(root, dest, numeric, out, report)


def write_dataset(info_root, output, dests, provenance, keep_jpegs=False):
    """Collect finished episode folders into one LeRobot dataset in the real dataset's format.

    Every folder must be finished and still hold its JPEGs, or nothing is written. The dataset
    is built beside `output` and replaces it only once complete, so a failure leaves any
    earlier output intact.
    """
    import pickle
    missing = [d.name for d in dests if not episode_ready(d)]
    if missing:
        raise SystemExit(f'{len(missing)} episode folders are unfinished or lost their images ({missing[:3]}...); '
                         f'nothing was written and {output} is unchanged')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = json.loads((info_root / 'meta/info.json').read_text())
    final, output = output, output.with_name(output.name + '.writing')
    shutil.rmtree(output, ignore_errors=True)
    task = __import__('pyarrow.parquet', fromlist=['x']).read_table(
        info_root / 'meta/tasks.parquet').to_pandas().index.tolist()[0]
    dataset = LeRobotDataset.create(repo_id='local/panthera_real_augmented', root=output, fps=info['fps'],
                                    robot_type=info.get('robot_type', 'panthera_ht_bimanual_real'),
                                    features=real_features(info_root), use_videos=False, image_writer_threads=0,
                                    metadata_buffer_size=20)
    reports, total = {}, 0
    try:
        for dest in dests:
            reports[dest.name] = json.loads((dest / 'done.json').read_text())
            with np.load(dest / 'numeric.npz') as numeric:
                n = len(numeric['action'])
                buffer = dataset.episode_buffer
                buffer.update(size=n, task=[task] * n, timestamp=np.arange(n) / info['fps'], frame_index=np.arange(n))
                for key in NUMERIC:
                    buffer[key] = numeric[key]
            for camera in CAMERAS:
                buffer[f'observation.images.{camera}'] = [str(dest / camera / f'{j:05d}.jpg') for j in range(n)]
            save_prepared_episode(dataset, pickle.loads((dest / 'image_stats.pkl').read_bytes()))
            total += n
    finally:
        dataset.finalize()
        dataset.stop_image_writer()
    (output / 'meta/provenance.json').write_text(json.dumps(dict(provenance, episodes=reports), indent=2,
                                                            default=str) + '\n')
    (output / 'COMPLETE').write_text(f'{len(dests)} episodes, {total} frames\n')
    shutil.rmtree(final, ignore_errors=True)
    output.rename(final)
    if not keep_jpegs:
        for dest in dests:
            for camera in CAMERAS:
                shutil.rmtree(dest / camera, ignore_errors=True)
    return total


def run_jobs(job, calls, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(job, *call) for call in calls]
        started = time.time()
        for k, future in enumerate(as_completed(futures), 1):
            future.result()
            elapsed = time.time() - started
            print(json.dumps(dict(rendered=k, of=len(calls), elapsed_s=round(elapsed),
                                  eta_h=round(elapsed / k * (len(calls) - k) / 3600, 2))), flush=True)


def export(args):
    revision = fetch(args.root, args.revision)
    info = json.loads((args.root / 'meta/info.json').read_text())
    episodes = list(range(info['total_episodes']))[:args.limit or None]
    rng = np.random.default_rng(args.seed)
    restyled = sorted(rng.choice(episodes, size=round(len(episodes) * args.fraction), replace=False).tolist()) \
        if args.fraction else []
    # (episode, restyle seed, realistic look, copy, wood cleanup); copies alternate realistic / wild when
    # balanced, and with --wood-cleanup both, cleanup alternates by episode and copy, so each demo has a
    # copy of each kind and each look gets half of each.
    clean = lambda e, k: {'on': True, 'off': False}.get(args.wood_cleanup, (e + k + k // 2) % 2 == 0)
    items = [(e, None, None, None, None) for e in episodes] if args.include_originals else []
    items += [(e, args.seed * 100003 + e + 1000003 * k, (k % 2 == 0) if args.balanced else None, k, clean(e, k))
              for k in range(args.copies) for e in restyled]
    settings = restyle_settings(args)
    contract = dict(source=REAL, revision=revision, items=items,
                    restyle={k: v for k, v in asdict(settings).items() if k != 'prompts'}, relight=args.relight,
                    scramble=args.scramble, masks='depth', soft_edges=args.soft_edges,
                    wood_cleanup=args.wood_cleanup, palettes=[REALISTIC_PALETTES, WILD_PALETTES])
    check_settings(args.work, contract)
    dests = [args.work / (f'episode_{e:04d}' + ('' if s is None else f'_restyled{k}')) for e, s, _, k, _ in items]
    if not args.write_finished and not args.render_only and output_matches(args.output, dests):
        print(f'{args.output} is already the complete dataset of these {len(dests)} episodes; nothing to do')
        return
    if stripped := reopen_stripped(dests):
        print(f'{len(stripped)} finished episodes lost their images to an earlier dataset write; rendering them again',
              flush=True)
    job = lambda i: (args.root, dests[i], items[i][0], items[i][1], settings, args.relight, args.scramble,
                     args.soft_edges, items[i][2], items[i][4])
    if args.write_finished:  # write what is rendered so far and stop (other pods may keep rendering)
        finished = [d for d in dests if episode_ready(d)]
        print(f'writing the {len(finished)} finished of {len(dests)} episodes', flush=True)
        provenance = dict(source=REAL, revision=revision, generator='tools/augment_real_dataset.py export',
                          partial=f'{len(finished)} of {len(dests)} episodes', work=str(args.work),
                          restyle={k: v for k, v in asdict(settings).items() if k != 'prompts'})
        total = write_dataset(args.root, args.output, finished, provenance, keep_jpegs=True)
        print(f'complete: {len(finished)} episodes, {total} frames at {args.output}')
        return
    todo = [i for i, d in enumerate(dests) if not (d / 'done.json').exists()]
    if args.reverse:  # a second pod works from the other end of the list
        todo = todo[::-1]
    print(f'{len(items) - len(todo)} of {len(items)} episodes ready; rendering {len(todo)} '
          f'({len(restyled) * args.copies} restyled copies)', flush=True)
    run_jobs(render_job, [job(i) for i in todo], args.workers)
    if args.render_only:
        print('rendering pass finished (--render-only); the main run writes the dataset', flush=True)
        return
    # Episodes another pod is still rendering: wait for them (taking over any that go stale).
    while missing := [i for i, d in enumerate(dests) if not (d / 'done.json').exists()]:
        print(f'waiting for {len(missing)} episodes rendered elsewhere', flush=True)
        time.sleep(60)
        run_jobs(render_job, [job(i) for i in missing], args.workers)
    provenance = dict(source=REAL, revision=revision, generator='tools/augment_real_dataset.py export',
                      method='table/background restyle; robot, parts and hands copied back from the real frames',
                      depth_model=DEPTH_MODEL, soft_edges=args.soft_edges, relight=args.relight,
                      include_originals=args.include_originals, restyled_source_episodes=restyled,
                      copies=args.copies, balanced=args.balanced, wood_cleanup=args.wood_cleanup,
                      work=str(args.work),
                      restyle={k: v for k, v in asdict(settings).items() if k != 'prompts'})
    total = write_dataset(args.root, args.output, dests, provenance, args.keep_jpegs)
    print(f'complete: {len(items)} episodes ({len(restyled) * args.copies} restyled), {total} frames at {args.output}')


def relight_job(source, index, maps_path, dest, look_seed):
    """One output episode of the lighting pass: the source episode as is, or relit."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    numeric, images = episode_rows(source, index)
    report = dict(source_index=index, frames=len(numeric['action']), relit=look_seed is not None)
    if look_seed is None:
        write_folder(source, dest, numeric, images, report)
        return
    frames = {camera: decode(images[camera]) for camera in CAMERAS}
    maps = load_maps(maps_path) if maps_path and Path(maps_path).exists() else compute_maps(frames)
    look = sample_lighting(look_seed)
    out = {camera: relight(frames[camera], maps[camera][1], look, maps[camera][0], camera) for camera in CAMERAS}
    report.update(lighting=look, look_seed=look_seed, maps='stored' if maps_path and Path(maps_path).exists()
                  else 'computed')
    write_folder(source, dest, numeric, out, report)


def relight_dataset(args):
    """Lighting and shadow changes (no diffusion) on a finished dataset: every source episode
    as is, plus `--variants` relit copies of a `--fraction` of them."""
    import pyarrow.parquet as pq
    try:  # an export still writing its dataset leaves the episode index without a parquet footer
        for f in args.source.glob('meta/episodes/*/*.parquet'):
            pq.read_metadata(f)
    except Exception:
        raise SystemExit(f'{args.source} is still being written (unreadable episode index); '
                         'wait for its COMPLETE file')
    provenance = json.loads((args.source / 'meta/provenance.json').read_text()) \
        if (args.source / 'meta/provenance.json').exists() else {}
    names = list(provenance.get('episodes', {}))
    info = json.loads((args.source / 'meta/info.json').read_text())
    count = info['total_episodes']
    if names and len(names) != count:
        raise SystemExit(f'{args.source}: provenance lists {len(names)} episodes, dataset has {count}')
    work = Path(provenance.get('work', '')) if args.maps is None else args.maps
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(count, size=round(count * args.fraction), replace=False).tolist())
    items = [(i, None) for i in range(count)] if args.include_source else []
    items += [(i, args.seed * 100003 + i + 1000003 * v) for v in range(args.variants) for i in chosen]
    maps_for = lambda i: str(work / names[i] / 'maps.npz') if names else None
    missing = sum(1 for i, s in items if s is not None and not (maps_for(i) and Path(maps_for(i)).exists()))
    print(f'{len(items)} output episodes ({len(items) - len(chosen) * args.variants} as is, '
          f'{len(chosen) * args.variants} relit; {missing} need depth and masks computed)', flush=True)
    complete = args.source / 'COMPLETE'
    check_settings(args.work, dict(source=str(args.source.resolve()), source_episodes=count,
                                   source_frames=info['total_frames'],
                                   source_complete=complete.read_text() if complete.exists() else None,
                                   maps=str(work), seed=args.seed, variants=args.variants, fraction=args.fraction,
                                   include_source=args.include_source))
    dests = [args.work / (f'episode_{i:04d}' + ('' if s is None else f'_lit{(s - args.seed * 100003 - i) // 1000003}'))
             for i, s in items]
    if output_matches(args.output, dests):
        print(f'{args.output} is already the complete dataset of these {len(dests)} episodes; nothing to do')
        return
    if stripped := reopen_stripped(dests):
        print(f'{len(stripped)} finished episodes lost their images to an earlier dataset write; rendering them again',
              flush=True)
    todo = [k for k, d in enumerate(dests) if not (d / 'done.json').exists()]
    run_jobs(relight_job, [(args.source, items[k][0], maps_for(items[k][0]), dests[k], items[k][1])
                           for k in todo], args.workers)
    total = write_dataset(args.source, args.output, dests,
                          dict(source=str(args.source), source_provenance={k: v for k, v in provenance.items()
                                                                           if k != 'episodes'},
                               generator='tools/augment_real_dataset.py relight', variants=args.variants,
                               fraction=args.fraction, include_source=args.include_source,
                               source_names=names), args.keep_jpegs)
    print(f'complete: {len(items)} episodes, {total} frames at {args.output}')


def preview(args):
    """A few frames per camera: original | restyle input | restyled | restyled + relit
    (with --no-restyle: original | kept pixels, blue = repainted). Same settings as export."""
    revision = fetch(args.root, args.revision)
    from dataclasses import replace
    from tools.sim2real_augment import Restyler
    settings = restyle_settings(args)
    depth_model = Depth()
    restyler = None if args.no_restyle else Restyler(settings)
    rows = []
    for k, episode in enumerate(args.episodes):
        _, images = episode_rows(args.root, episode)
        seed = args.seed + k
        for camera in CAMERAS:
            frames = decode(images[camera])
            pick = np.linspace(0, len(frames) - 1, args.frames + 2)[1:-1].astype(int).tolist()
            shown = frames[pick]
            disparity = depth_model(shown)
            keep = object_masks(shown, disparity)
            if restyler is None:
                overlay = shown.copy()
                overlay[~keep] = (overlay[~keep] * .3 + np.array([0, 80, 255]) * .4).astype(np.uint8)
                columns = [shown, overlay]
            else:
                source = scramble_background(shown, keep, seed)[0] if args.scramble else shown
                restyler.settings = replace(settings, prompts=prompts(is_realistic(seed)))
                restyled = restyler.episode(source, None, None, keep, seed, camera, ~keep,
                                            controls=controls(shown, disparity, keep))
                if args.soft_edges:
                    restyled = soft_composite(shown, restyled, keep)
                columns = [shown, source, restyled, relight(restyled, disparity, sample_lighting(seed), keep, camera)]
            for i in range(len(shown)):
                rows.append(np.concatenate([c[i] for c in columns], 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    names = 'original, kept (blue = repainted)' if restyler is None else \
        'original, restyle input' + (' (scrambled)' if args.scramble else '') + ', restyled, restyled + relit'
    print(f'wrote {args.output} (revision {revision[:8]}): columns {names}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--root', type=Path, default=ROOT / 'data/real_gear', help='local copy of the real dataset')
    common.add_argument('--revision', help='dataset revision (default: latest)')
    common.add_argument('--seed', type=int, default=0)
    common.add_argument('--batch', type=int, default=8)
    common.add_argument('--steps', type=int, default=4)
    common.add_argument('--size', type=int, default=384, help='diffusion resolution (frames are 256 px)')
    common.add_argument('--strength', type=float, default=.95)
    common.add_argument('--guidance', type=float, default=1.)
    common.add_argument('--depth-scale', type=float, default=.6)
    common.add_argument('--canny-scale', type=float, default=.6)
    common.add_argument('--hard-edges', dest='soft_edges', action='store_false',
                        help='copy kept pixels back with a hard mask edge instead of a guided-filter blend')
    common.add_argument('--no-scramble', dest='scramble', action='store_false',
                        help='start the restyle from the original table instead of a random colour/texture')
    p = sub.add_parser('export', parents=[common])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True, help='per-episode render folders (resumable)')
    p.add_argument('--fraction', type=float, default=1., help='share of episodes that get restyled copies')
    p.add_argument('--copies', type=int, default=2, help='restyled copies per chosen episode')
    p.add_argument('--wood-cleanup', choices=('both', 'on', 'off'), default='both',
                   help='repaint leftover wood beside kept objects: on, off (the earlier masks), or both '
                        '(alternating, so each demo has a copy of each)')
    p.add_argument('--unbalanced', dest='balanced', action='store_false',
                   help='pick each copy\'s look at random instead of alternating realistic / wild')
    p.add_argument('--no-originals', dest='include_originals', action='store_false')
    p.add_argument('--relight', action='store_true',
                   help='also apply the lighting change here (default: leave it to the `relight` pass)')
    p.add_argument('--reverse', action='store_true',
                   help='render from the end of the list (a second pod sharing the same --work folder)')
    p.add_argument('--render-only', action='store_true',
                   help='stop after rendering; the main run (without this flag) writes the dataset')
    p.add_argument('--write-finished', action='store_true',
                   help='only write a dataset from the episodes finished so far (keeps their JPEGs, so a '
                        'running render is not disturbed)')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--keep-jpegs', action='store_true')
    p = sub.add_parser('relight', help='lighting/shadow variants of a finished dataset, no diffusion')
    p.add_argument('--source', type=Path, required=True, help='a dataset written by `export` (or the real one)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True, help='per-episode folders (resumable)')
    p.add_argument('--maps', type=Path, help='export --work folder with maps.npz (default: from provenance)')
    p.add_argument('--variants', type=int, default=1, help='relit copies per chosen episode')
    p.add_argument('--fraction', type=float, default=1.)
    p.add_argument('--no-source', dest='include_source', action='store_false',
                   help='only the relit copies, not the source episodes as they are')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--keep-jpegs', action='store_true')
    p = sub.add_parser('preview', parents=[common])
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/real_aug_preview.jpg')
    p.add_argument('--episodes', type=int, nargs='+', default=[0, 101])
    p.add_argument('--frames', type=int, default=2)
    p.add_argument('--no-restyle', action='store_true', help='masks only (no diffusion models)')
    args = parser.parse_args()
    {'export': export, 'preview': preview, 'relight': relight_dataset}[args.command](args)


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
