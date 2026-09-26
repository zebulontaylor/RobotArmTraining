#!/usr/bin/env python3
"""Roll out the fine-tuned VLA-Adapter policy in the Panthera MuJoCo scene.

The checkpoint may be an extracted training directory, the ``.tar.gz`` made by
the training notebook, or a Google Drive ``.zip`` download.  Archives are
selectively unpacked: optimizer state is deliberately skipped because it is
not needed for inference.

This script expects the pinned VLA-Adapter checkout and its Python environment;
see the README's "Model rollout" section for the one-time setup.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
VLA_COMMIT = "23fa0c9c159e2aa04341cdd3e924f44061311060"
VLA_REPOSITORY = "https://github.com/OpenHelix-Team/VLA-Adapter.git"
BASE_MODEL_REPOSITORY = "Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b"
BASE_CHECKPOINT = "checkpoints/step-020792-epoch-01-loss=0.5268.pt"
UNNORM_KEY = "robot_arm_learning_panthera"
ROLLOUT_START_Z = 0.20  # Metres above the world origin.
REQUIRED_CHECKPOINT_FILES = (
    "dataset_statistics.json",
    "action_head--latest_checkpoint.pt",
    "proprio_projector--latest_checkpoint.pt",
    "trainable_extras--latest_checkpoint.pt",
    "lora_adapter/adapter_config.json",
    "lora_adapter/adapter_model.safetensors",
)


def existing_vla_venv() -> Path | None:
    """Locate the training environment already used by this project."""
    configured = os.environ.get("VLA_VENV")
    candidates = [
        Path(configured).expanduser() if configured else None,
        REPO_ROOT / ".venv",
        REPO_ROOT / "venv",
        Path.home() / "venvs" / "vla-adapter",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "bin" / "python").is_file():
            return candidate.resolve()
    return None


def reexec_in_vla_venv() -> None:
    """Restart under the existing VLA venv when invoked with plain python."""
    venv = existing_vla_venv()
    if venv is None or Path(sys.prefix).resolve() == venv:
        return
    if os.environ.get("ROBOT_ARM_ROLLOUT_REEXEC") == "1":
        return
    python = venv / "bin" / "python"
    print(f"Restarting with the existing VLA environment: {python}", flush=True)
    environment = os.environ.copy()
    environment["ROBOT_ARM_ROLLOUT_REEXEC"] = "1"
    os.execve(
        str(python),
        [str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def default_vla_dir() -> Path:
    local = REPO_ROOT / "VLA-Adapter"
    if local.is_dir():
        return local
    return Path.home() / ".cache" / "robot-arm-learning" / "VLA-Adapter"


def default_checkpoint() -> Path:
    """Return the newest plausible artifact in Downloads/Downlods."""
    candidates: list[Path] = []
    for dirname in ("Downloads", "Downlods"):
        root = Path.home() / dirname
        if not root.is_dir():
            continue
        candidates.extend(p for p in root.glob("robot-arm-learning*") if p.is_dir())
        candidates.extend(root.glob("robot-arm-learning*.zip"))
        candidates.extend(root.glob("robot-arm-learning*.tar.gz"))
    if not candidates:
        raise SystemExit(
            "No robot-arm-learning checkpoint found in ~/Downloads (or ~/Downlods). "
            "Pass it explicitly with --checkpoint."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _archive_root(names: list[str]) -> str:
    matches = [name for name in names if name.endswith("/dataset_statistics.json")]
    if len(matches) != 1:
        raise SystemExit(
            "Expected exactly one dataset_statistics.json in the checkpoint archive; "
            f"found {len(matches)}."
        )
    return matches[0][: -len("dataset_statistics.json")]


def materialize_checkpoint(source: Path, cache_root: Path) -> Path:
    source = source.expanduser().resolve()
    if source.is_dir():
        if (source / "dataset_statistics.json").is_file():
            missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (source / name).is_file()]
            if missing:
                raise SystemExit(f"Checkpoint directory is missing: {', '.join(missing)}")
            return source
        children = [p.parent for p in source.rglob("dataset_statistics.json")]
        if len(children) == 1:
            return children[0]
        raise SystemExit(f"Could not identify a checkpoint directory under {source}")
    if not source.is_file():
        raise SystemExit(f"Checkpoint does not exist: {source}")

    fingerprint = hashlib.sha256(
        f"{source}:{source.stat().st_size}:{source.stat().st_mtime_ns}".encode()
    ).hexdigest()[:12]
    output = cache_root / "checkpoints" / f"{source.name}-{fingerprint}"
    complete = output / ".inference-files-complete"
    if complete.is_file() and all((output / name).is_file() for name in REQUIRED_CHECKPOINT_FILES):
        return output

    print(f"Extracting inference weights from {source} to {output}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            root = _archive_root(archive.namelist())
            for relative in REQUIRED_CHECKPOINT_FILES:
                member = root + relative
                try:
                    info = archive.getinfo(member)
                except KeyError as exc:
                    raise SystemExit(f"Checkpoint archive is missing {relative}") from exc
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as reader, destination.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
    elif tarfile.is_tarfile(source):
        with tarfile.open(source, "r:*") as archive:
            names = archive.getnames()
            root = _archive_root(names)
            for relative in REQUIRED_CHECKPOINT_FILES:
                member = archive.getmember(root + relative)
                reader = archive.extractfile(member)
                if reader is None:
                    raise SystemExit(f"Checkpoint archive is missing {relative}")
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with reader, destination.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
    else:
        raise SystemExit(f"Unsupported checkpoint archive: {source}")
    complete.write_text("ok\n")
    return output


def ensure_vla_checkout(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        print(f"Cloning the pinned VLA-Adapter source into {path}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", VLA_REPOSITORY, str(path)], check=True)
        subprocess.run(["git", "checkout", VLA_COMMIT], cwd=path, check=True)
    marker = path / "prismatic" / "extern" / "hf" / "modeling_prismatic.py"
    if not marker.is_file():
        raise SystemExit(f"Not a VLA-Adapter checkout: {path}")
    return path


def require_ml_dependencies(vla_dir: Path) -> None:
    sys.path.insert(0, str(vla_dir))
    missing = []
    for package in ("transformers", "peft", "timm", "tokenizers"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit(
            "The VLA inference environment is not installed (missing "
            + ", ".join(missing)
            + "). Follow the one-time setup in README.md, then run this script "
              "with .venv-rollout/bin/python."
        )


def ensure_base_model(cache_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not (path / "config.json").is_file() or not (path / BASE_CHECKPOINT).is_file():
            raise SystemExit(f"Base model is incomplete: {path}")
        return path
    path = cache_root / "base-model"
    if not (path / "config.json").is_file() or not (path / BASE_CHECKPOINT).is_file():
        print(f"Downloading base model {BASE_MODEL_REPOSITORY} (about 2.5 GiB)", flush=True)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=BASE_MODEL_REPOSITORY,
            local_dir=path,
            allow_patterns=["config.json", BASE_CHECKPOINT],
        )
    return path


def strip_ddp(state: dict) -> dict:
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_policy(vla_dir: Path, base_dir: Path, checkpoint: Path, device_name: str):
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.load import load
    from prismatic.models.projectors import ProprioProjector

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    dtype = torch.bfloat16
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        raise SystemExit("VLA-Adapter inference in bfloat16 needs an Ampere-or-newer GPU")

    config_dir = vla_dir / "pretrained_models" / "configs"
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)
    processor = AutoProcessor.from_pretrained(config_dir, trust_remote_code=False)

    print("Loading base vision-language model...", flush=True)
    # The native checkpoint contains every base weight. Inference mode builds
    # the Qwen module from config before restoring it, avoiding an unnecessary
    # second Qwen download and the training-only FlashAttention requirement.
    base_vlm = load(base_dir, hf_token="", load_for_training=False)
    replacements = (
        ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
        ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
        ("llm_backbone.llm", "language_model"),
        ("projector.projector.0", "projector.fc1"),
        ("projector.projector.2", "projector.fc2"),
        ("projector.projector.4", "projector.fc3"),
        ("gamma", "scale_factor"),
    )
    renamed = {}
    for key, value in base_vlm.state_dict().items():
        for old, new in replacements:
            key = key.replace(old, new)
        renamed[key] = value

    config = AutoConfig.from_pretrained(config_dir)
    model = AutoModelForVision2Seq.from_config(config, torch_dtype=dtype)
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected base-model tensors: {unexpected[:8]}")
    # The HF wrapper owns action-query parameters that the native base VLM lacks.
    if any("action_queries" not in key for key in missing):
        print(f"Warning: base conversion left {len(missing)} tensors uninitialized", flush=True)
    del renamed, base_vlm
    gc.collect()

    model.vision_backbone.set_num_images_in_input(2)
    model = PeftModel.from_pretrained(model, checkpoint / "lora_adapter", is_trainable=False)
    extras = strip_ddp(torch.load(
        checkpoint / "trainable_extras--latest_checkpoint.pt",
        weights_only=True,
        map_location="cpu",
    ))
    load_result = model.load_state_dict(extras, strict=False)
    if load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected trained extras: {load_result.unexpected_keys}")

    core = model.base_model.model
    with (checkpoint / "dataset_statistics.json").open() as stream:
        core.norm_stats = json.load(stream)
    model.norm_stats = core.norm_stats

    action_head = L1RegressionActionHead(
        input_dim=core.llm_dim,
        hidden_dim=core.llm_dim,
        action_dim=7,
        use_pro_version=True,
    )
    action_head.load_state_dict(strip_ddp(torch.load(
        checkpoint / "action_head--latest_checkpoint.pt",
        weights_only=True,
        map_location="cpu",
    )))
    proprio_projector = ProprioProjector(llm_dim=core.llm_dim, proprio_dim=8)
    proprio_projector.load_state_dict(strip_ddp(torch.load(
        checkpoint / "proprio_projector--latest_checkpoint.pt",
        weights_only=True,
        map_location="cpu",
    )))

    model = model.to(device=device, dtype=dtype).eval()
    action_head = action_head.to(device=device, dtype=dtype).eval()
    proprio_projector = proprio_projector.to(device=device, dtype=dtype).eval()
    return SimpleNamespace(
        model=model,
        processor=processor,
        action_head=action_head,
        proprio_projector=proprio_projector,
        device=device,
        dtype=dtype,
        stats=core.norm_stats[UNNORM_KEY],
    )


def normalize_proprio(state: np.ndarray, stats: dict) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.where(mask, 2.0 * (state - low) / (high - low + 1e-8) - 1.0, state)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def prepare_image(image: np.ndarray):
    """Match the JPEG, resize, and center-crop path used by VLA evaluation."""
    import cv2
    from PIL import Image

    # The official evaluator round-trips live observations through JPEG before
    # resizing, matching the JPEG-backed RLDS training examples.
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not ok:
        raise RuntimeError("Could not JPEG-encode a policy observation")
    decoded = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(decoded).resize((224, 224), Image.Resampling.LANCZOS)

    height, width = pil.height, pil.width
    crop = math.sqrt(0.9)
    crop_h, crop_w = round(height * crop), round(width * crop)
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    return pil.crop((left, top, left + crop_w, top + crop_h)).resize(
        (224, 224), Image.Resampling.BILINEAR
    )


def predict_actions(policy, shoulder: np.ndarray, wrist: np.ndarray,
                    state: np.ndarray, instruction: str) -> np.ndarray:
    import torch

    prompt = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        "<|im_end|>\n<|im_start|>user\nWhat action should the robot take to "
        f"{instruction.lower()}?<|im_end|>\n<|im_start|>assistant\n"
    )
    primary = policy.processor(prompt, prepare_image(shoulder)).to(
        policy.device, dtype=policy.dtype
    )
    wrist_inputs = policy.processor(prompt, prepare_image(wrist)).to(
        policy.device, dtype=policy.dtype
    )
    primary["pixel_values"] = torch.cat(
        [primary["pixel_values"], wrist_inputs["pixel_values"]], dim=1
    )
    proprio = normalize_proprio(state, policy.stats["proprio"])
    with torch.inference_mode():
        actions, _ = policy.model.predict_action(
            **primary,
            unnorm_key=UNNORM_KEY,
            do_sample=False,
            proprio=proprio,
            proprio_projector=policy.proprio_projector,
            action_head=policy.action_head,
            use_film=False,
        )
    return np.asarray(actions, dtype=np.float64)


def rotvec_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    quat = np.zeros(4)
    if angle < 1e-12:
        quat[0] = 1.0
    else:
        import mujoco

        mujoco.mju_axisAngle2Quat(quat, rotvec / angle, angle)
    return quat


def current_state(sim) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    pos, quat = sim.ee_pose()
    euler = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz")
    finger_opening = float(sim.data.ctrl[sim.grip_act])
    # POS_EULER is xyz + roll/pitch/yaw + one padding value + gripper.
    return np.concatenate((pos, euler, [0.0, finger_opening])).astype(np.float32)


def display_frame(shoulder: np.ndarray, wrist: np.ndarray, step: int,
                  instruction: str, last_action: np.ndarray | None) -> np.ndarray:
    import cv2

    frame = np.concatenate((shoulder, wrist), axis=1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 43), (0, 0, 0), -1)
    cv2.putText(frame, f"step {step:04d}  {instruction}  [r] reset  [q] quit", (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if last_action is not None:
        text = "action " + np.array2string(last_action, precision=3, suppress_small=True)
        cv2.putText(frame, text, (8, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (90, 240, 90), 1, cv2.LINE_AA)
    return frame


def rollout(args, policy) -> None:
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    import cv2
    import mujoco

    sys.path.insert(0, str(REPO_ROOT / "sim"))
    sys.path.insert(0, str(REPO_ROOT / "teleop"))
    from panthera_env import PantheraSim
    from render_vla_dataset import shoulder_camera, wrist_camera

    sim = PantheraSim(dynamics=args.dynamics)

    def reset_scene(seed: int) -> tuple[np.ndarray, np.ndarray]:
        sim.reset(
            randomize=not args.fixed_scene,
            rng=np.random.default_rng(seed),
        )
        start_pos, start_quat = sim.ee_pose()
        start_pos[2] = ROLLOUT_START_Z
        start_q, pos_error, rot_error = sim.ik(
            start_pos,
            start_quat,
            q_init=sim.q,
            max_joint_step=None,
        )
        if pos_error > 0.005 or rot_error > math.radians(1.0):
            raise RuntimeError(
                "Could not lower the rollout start pose "
                f"(IK error {pos_error * 1000:.1f} mm, "
                f"{math.degrees(rot_error):.1f} deg)"
            )
        sim.data.qpos[sim.arm_qadr] = start_q
        sim.data.qvel[sim.arm_dofadr] = 0.0
        sim.set_arm_ctrl(start_q, immediate=True)
        mujoco.mj_forward(sim.model, sim.data)
        return sim.ee_pose()

    target_pos, target_quat = reset_scene(args.seed)
    renderers = (
        mujoco.Renderer(sim.model, height=256, width=256),
        mujoco.Renderer(sim.model, height=256, width=256),
    )
    cameras = (shoulder_camera(sim.model), wrist_camera(sim.model))
    writer = None
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), args.hz, (512, 256)
        )
        if not writer.isOpened():
            raise SystemExit(f"Could not open video output {args.video}")

    physics_steps = max(1, round((1.0 / args.hz) / sim.dt))
    action_queue: list[np.ndarray] = []
    last_action = None
    # Actions are frame-to-frame pose deltas.  Integrate them into a commanded
    # pose rather than repeatedly applying them to the measured pose: the
    # latter turns gravity/servo tracking error into a new target every tick
    # and makes even an all-zero action sequence drift downward.
    limit = f"{args.steps} control steps" if args.steps else "until you quit"
    print(
        f"Rolling out '{args.instruction}' {limit} at {args.hz:g} Hz "
        f"(seed {args.seed}; r resets, q quits)",
        flush=True,
    )
    step = 0
    reset_count = 0
    try:
        while not args.steps or step < args.steps:
            tick_started = time.perf_counter()
            images = []
            for renderer, camera in zip(renderers, cameras):
                renderer.update_scene(sim.data, camera)
                images.append(renderer.render().copy())

            if not action_queue:
                started = time.perf_counter()
                chunk = predict_actions(
                    policy, images[0], images[1], current_state(sim), args.instruction
                )
                action_queue.extend(chunk[:args.open_loop])
                print(
                    f"policy query at step {step}: {len(action_queue)} actions in "
                    f"{time.perf_counter() - started:.2f}s",
                    flush=True,
                )

            action = np.asarray(action_queue.pop(0), dtype=np.float64)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(f"Policy returned invalid action: {action}")
            action[:6] *= args.action_scale
            target_pos = target_pos + action[:3]
            delta_quat = rotvec_quat(action[3:6])
            next_target_quat = np.zeros(4)
            mujoco.mju_mulQuat(next_target_quat, delta_quat, target_quat)
            mujoco.mju_normalize4(next_target_quat)
            target_quat = next_target_quat
            q_target, pos_error, rot_error = sim.ik(
                target_pos, target_quat, q_init=sim.q, max_joint_step=args.max_joint_step
            )
            sim.set_arm_ctrl(q_target)
            sim.set_gripper(float(action[6]))
            sim.step(physics_steps)
            last_action = action

            frame = display_frame(images[0], images[1], step, args.instruction, last_action)
            if writer is not None:
                writer.write(frame)
            key = -1
            if not args.no_display:
                cv2.imshow("Panthera VLA rollout - shoulder | wrist", frame)
                elapsed = time.perf_counter() - tick_started
                delay_ms = 1
                if not args.no_realtime:
                    delay_ms = max(1, round((1.0 / args.hz - elapsed) * 1000))
                key = cv2.waitKey(delay_ms) & 0xFF
            elif not args.no_realtime:
                remaining = 1.0 / args.hz - (time.perf_counter() - tick_started)
                if remaining > 0:
                    time.sleep(remaining)
            print(
                f"step {step:04d}  xyz={sim.ee_pos().round(3)}  grip={action[6]:.2f}  "
                f"ik=({pos_error * 1000:.1f}mm, {math.degrees(rot_error):.1f}deg)",
                flush=True,
            )
            if key == ord("q"):
                break
            if key == ord("r"):
                reset_count += 1
                reset_seed = args.seed + reset_count
                target_pos, target_quat = reset_scene(reset_seed)
                action_queue.clear()
                last_action = None
                step = 0
                print(f"Simulation reset (seed {reset_seed})", flush=True)
                continue
            step += 1
    finally:
        if writer is not None:
            writer.release()
        for renderer in renderers:
            renderer.close()
        cv2.destroyAllWindows()
    if args.video:
        print(f"Saved rollout video to {args.video.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="checkpoint directory, .zip, or .tar.gz")
    parser.add_argument(
        "--vla-dir", type=Path,
        default=Path(os.environ.get("VLA_ADAPTER_DIR", default_vla_dir())),
        help="pinned VLA-Adapter source checkout",
    )
    parser.add_argument("--base-model", type=Path, help="already-downloaded base model directory")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path.home() / ".cache" / "robot-arm-learning")
    parser.add_argument("--instruction", default="stack the three colored cubes")
    parser.add_argument(
        "--steps", type=int, default=0,
        help="control steps before stopping; 0 (default) runs until q",
    )
    parser.add_argument("--dynamics", choices=("contact-v2", "weld-v1"), default="contact-v2")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--open-loop", type=int, default=8, choices=range(1, 9), metavar="1..8")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fixed-scene", action="store_true", help="use the XML cube layout")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--max-joint-step", type=float, default=0.15)
    parser.add_argument("--video", type=Path, help="optional MP4 output")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--no-realtime", action="store_true",
        help="run as fast as possible instead of pacing control steps",
    )
    parser.add_argument("--mujoco-gl", default="egl", choices=("egl", "glfw", "osmesa"))
    args = parser.parse_args()
    if args.steps < 0 or args.hz <= 0 or args.action_scale <= 0:
        parser.error("--steps must be nonnegative; --hz and --action-scale must be positive")
    if args.no_display and args.steps == 0:
        parser.error("--no-display requires a positive --steps value")
    return args


def main() -> None:
    reexec_in_vla_venv()
    args = parse_args()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    checkpoint_source = args.checkpoint or default_checkpoint()
    print(f"Checkpoint artifact: {checkpoint_source}", flush=True)
    vla_dir = ensure_vla_checkout(args.vla_dir)
    require_ml_dependencies(vla_dir)
    checkpoint = materialize_checkpoint(checkpoint_source, args.cache_dir)
    local_base = vla_dir / "pretrained_models" / "prism-qwen25-extra-dinosiglip-224px-0_5b"
    base_override = args.base_model
    if base_override is None and (local_base / "config.json").is_file() \
            and (local_base / BASE_CHECKPOINT).is_file():
        base_override = local_base
        print(f"Using existing base model: {local_base}", flush=True)
    base_dir = ensure_base_model(args.cache_dir, base_override)
    policy = load_policy(vla_dir, base_dir, checkpoint, args.device)
    rollout(args, policy)


if __name__ == "__main__":
    main()
