# Autonomous pin regrasp collection

Collect camera observations and actions for approaching and grasping an upright
pin in MuJoCo. Each attempt starts with a randomized, table-supported grasp,
opens the fingers, retreats, returns along measured retreat poses, closes,
and lifts the pin. The return images come from executing the return motion.

The scene uses the two Panthera arms in the
[photograph setup profile](../../config/real_setup_20261027.json). The left arm manipulates the pin while the
right arm holds the large carrier. Reset initializes the arms and parts, then
establishes both grasps through finger contact. Every recorded motion runs the
position servos and contact physics; the objects are not welded to the grippers.

## Run a collection

From the repository root, start a continuous run that repositions its own pin:

```bash
MUJOCO_GL=egl .venv-assembly/bin/python tools/collect_continuous_pin.py \
  --output outputs/pin_regrasp/continuous_run --cycles 5 --image-size 320
```

The controller initializes the scene once. It releases, retreats, approaches,
grasps and lifts the pin, then carries it to another sampled tabletop position
and lowers it before the next cycle. The right arm keeps gripping the carrier.
There are no scene resets or object-pose assignments between cycles. A lost
grasp, failed placement, or controller error stops the run rather than silently
resetting the scene. The initial supported grasp still requires setup on real
hardware.

The continuous controller uses quintic easing for acceleration and deceleration,
curved retreats, gradual finger motion, and sampled movement speeds. Default
retreats span 30–90 mm in a 12° cone around the recorded direction. Wrist
perturbations and placement yaw vary within ±6°, and subsequent placements vary
within ±25 mm in each horizontal axis around the initial position. Placement
yaw rotates the held pin about world vertical so it can return upright to the
table. The next grasp uses the new pose.

Each approach samples a starting finger position from 12–25 mm per finger and
an arrival position from 3.5–6 mm per finger. The fingers begin closing after
45% of the approach time, while the tool is still moving toward the pin. They
retain clearance at arrival, then finish gripping during a short 0.35 s close
phase. Both sampled openings are saved in the cycle metadata. Set
`approach_start_min_m`, `approach_start_max_m`, `approach_arrival_min_m`,
`approach_arrival_max_m`, and `approach_close_start` in the continuous JSON
configuration to adjust the ranges and timing.

`continuous.mp4` shows every observed control frame in order, without episode
cuts, title holds, or playback speed changes. `trajectory.npz` retains numeric
observations, actions, phases, cycle IDs and training masks; `summary.json`
records completed cycles and any failure. Native camera videos preserve the
camera observations separately from the annotated preview using lossy H.264
CRF 19 encoding. `frames.jsonl` maps each video frame to the numeric observation
index and simulation timestamp. Use entire runs as
train/validation groups because their cycles share the same scene and grasp.

For independent randomized episodes instead of physical repositioning, use the
command below.

From the repository root, use a Python environment with
[`requirements-actuator.txt`](../../requirements-actuator.txt) installed:

```bash
MUJOCO_GL=egl .venv-assembly/bin/python tools/collect_pin_regrasps.py \
  --output outputs/pin_regrasp/first_run \
  --episodes 1000 --max-attempts 1500 --workers 4 --image-size 256
```

`MUJOCO_GL=egl` selects headless rendering on machines with EGL support. Omit it
when using the machine's window-system renderer. `--no-images` runs state-only
collection for controller diagnostics. `--workers` runs independent simulator
processes; reduce it if renderer memory limits the batch.

The command writes successful episodes incrementally and stops at the requested
success count. An attempt limit prevents an endless retry loop; exhausting it
returns exit code 1 and preserves the data already collected. Use an empty output
directory for each run. `--seed` sets the first attempt seed, with successive
attempts using successive seeds. Use disjoint seed ranges for separate datasets.

Each attempt varies:

| Parameter | Default sampling |
| --- | --- |
| Pin XY | Independently within ±20 mm of a recorded pickup position mapped to the left arm's simulation base |
| Pin yaw | Uniformly within ±180°; the round pin is nearly yaw-symmetric |
| Grasp orientation | Recorded pickup rotation with ±3° local rotation-vector perturbations on each axis |
| Retreat distance | Uniformly from 20 to 60 mm |
| Retreat direction | Within an 8° cone around the reverse of the recorded descent direction |
| Retreat wrist rotation | Up to ±3° on each local rotation-vector axis, smoothly applied and reversed during return |

The [pickup profile](pin_regrasp_profile.json) contains 20 paired poses from real
October 5 recordings. Its setup-specific selection finds a later left gripper
closure at the loose pin location and the preceding descent. It is separate
from the insertion reset profile. The median recorded left pitch is about 46°
downward. Each sample retains its source episode, timestamp, tool orientations,
and paired right-arm state. Base extrinsics are approximate; table height and
grasp positions are retargeted for the simulated parts.

Pin placement and heading ranges are defined in
[`MujocoPinBackend.reset`](../pin_regrasp_backend.py). Camera placement, lighting,
and object shape are fixed. This dataset covers local regrasping, not arbitrary
clutter or insertion.

To change motion settings, pass `--config path/to/settings.json`. Unspecified
fields retain their defaults. For example:

```json
{
  "retreat_min_m": 0.02,
  "retreat_max_m": 0.06,
  "speed_m_s": 0.025,
  "release_s": 0.5,
  "close_s": 0.7,
  "verify_s": 0.3,
  "lift_m": 0.03,
  "retreat_cone_deg": 8.0,
  "rotation_jitter_deg": 3.0,
  "angular_speed_deg_s": 15.0
}
```

The simulator rejects failed IK, excessive tracking error, and unexpected
penetrating contacts. These are execution checks, not a full collision-free
path planner. Unreachable samples become failed attempts. A successful episode
must establish loaded contact on both finger pads, then retain that contact
while the pin remains at least 60% of the requested lift above its reset height
for every verification sample. The right arm must retain its carrier grasp
throughout collection. Return trajectories retrace both measured position and
orientation; interpolation bounds Cartesian and angular command speeds.

## Dataset format

`config.json` describes the action layout and collection settings.
`attempts.jsonl` records each completed attempt's seed, reset parameters, outcome,
failure reason, and successful episode path. `summary.json` holds aggregate
counts and completion status. Each successful episode has an NPZ archive and a
JSON metadata file under `episodes/`. `--save-failures` also writes rejected
trajectories under `failures/`; those have no training labels enabled.

For an episode containing T actions:

| Array | Shape and meaning |
| --- | --- |
| `actions` | `(T, 13)`: world XYZ in metres, row-major 3×3 target rotation, per-finger opening in metres |
| `obs_time` | `(T+1,)`: simulated seconds |
| `obs_state` | `(T+1, 28)`: left then right; each arm has six measured angles in radians, six velocities in rad/s, two signed finger positions in metres |
| `obs_ee_position`, `obs_ee_rotation` | Measured world tool position and rotation |
| `obs_image_overhead`, `obs_image_left_wrist`, `obs_image_right_wrist` | `(T+1, H, W, 3)` RGB uint8 images, unless disabled |
| `obs_image_overview` | External scene view for inspecting both arms; not a real robot camera |
| `obs_object_pose` | Pin world XYZ followed by quaternion WXYZ; privileged validation state |
| `obs_grasped` | Physical bilateral pad-contact flag; privileged validation state |
| `obs_right_grasped`, `obs_right_ee_position`, `obs_right_ee_rotation` | Carrier grip flag and measured right-arm tool pose |
| `obs_carrier_pose` | Carrier world XYZ and quaternion WXYZ, for retention and drift checks |
| `phases`, `phase_names` | Per-action indices and their string lookup table |
| `training_mask` | True for return/close actions in successful episodes |

Observation `i` precedes action `i`; observation `i+1` follows it. The actual
control frequency is quantized to the 1 ms physics timestep and saved as
`reset.control_hz` in episode metadata. At requested 30 Hz it is about 30.303 Hz.

Load with `np.load(path, allow_pickle=False)`. Select actions using
`training_mask` and pair them with the corresponding pre-action observations.
Release/retreat actions prepare the task; lift/verify actions score it. The mask
excludes both groups. Keep whole episodes together when splitting train and
validation data. Do not use `obs_object_pose` or `obs_grasped` as policy inputs
for a camera/proprioception policy intended for hardware. These files need an
adapter before use with trainers expecting the repository's joint-action or
LeRobot formats.

## Render a video

From the repository root, render stored frames without rerunning simulation:

```bash
.venv-assembly/bin/python tools/render_pin_regrasp_video.py \
  --input outputs/pin_regrasp/first_run \
  --output outputs/pin_regrasp/first_run/preview.mp4 --limit 3
```

This requires `ffmpeg`. The H.264 video plays observations in recorded time,
shows the scene and wrist views, labels each phase, and reports the outcome at
the end of each episode. Its adjacent manifest lists the source episodes.

## Port the collector to hardware

The state machine in
[`collection/pin_regrasp.py`](../../collection/pin_regrasp.py) depends only on the
`PinBackend` protocol. Implement another backend and call
`collect_episode(backend, seed, config)`; the supplied CLI instantiates MuJoCo
only.

The hardware backend must implement:

1. `reset(seed)`: physically position and localize a supported pin, establish a
   verified grasp while the right arm holds the carrier, and return the measured
   reset parameters. Include `approach_retreat_direction` in the same workspace
   frame to constrain sampling around the real approach. Simulation's pose
   assignment must become a robot placement routine or fixture reset.
2. `observe()`: return synchronized camera frames, robot state, tool pose, and
   timestamps. Use the same calibrated workspace frame throughout an episode.
3. `command(position, rotation, opening)`: execute one timed control interval
   through the robot's IK, joint/rate limits, workspace and collision checks.
   Convert per-finger opening to the hardware's gripper command units here.
4. `grasped()` and `verify_lift(min_height)`: measure grasp retention and object
   lift using available sensors. A closed-gripper command or elevated tool pose
   alone does not establish success.
5. `stop()` and `close()`: hold/stop motion and release interface resources.

Continuous collection also calls `supported()` before release and after
placement. The hardware adapter must detect that the pin is supported by the
table; tool height alone does not establish support. Placement motion uses the
same calibrated workspace frame and motion checks as the approach.

The hardware backend must reject unsafe motion before execution and raise
`RuntimeError` for expected failed attempts. Other exceptions propagate after
the collector calls `stop()`. The simulation backend is a reference
implementation, not a calibrated hardware controller.

Run the focused tests from the repository root:

```bash
.venv-assembly/bin/python -m pytest -q \
  tests/test_pin_regrasp_collection.py \
  tests/test_collect_pin_regrasps_cli.py \
  tests/test_pin_regrasp_backend.py
```
