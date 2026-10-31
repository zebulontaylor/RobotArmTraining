"""Extract tabletop pin pickups from the original October 5 recordings."""
import argparse
import json
from pathlib import Path

import numpy as np


def build_profile(recordings, annotations, limit=20):
    samples = []
    for entry in json.loads(annotations.read_text())['episodes']:
        episode = recordings / entry['id']
        metadata = json.loads((episode / 'metadata.json').read_text())
        if not metadata.get('complete') or metadata.get('label') != 'kept':
            continue
        rows = [json.loads(line) for line in (episode / 'states.jsonl').open()]
        for i, row in enumerate(rows):
            t = row['episode_time_s']
            arm = row['arms']['left']
            position = arm['tool_pose_base']['position_m']
            # In this fixed recording setup, the loose pin lies farther forward
            # than the gear/carrier. The later open-to-closed event separates its
            # pickup from gear pickup and the final insertion release.
            if t < 8 or t > entry['end_s'] - 2 or position[0] < .37 or not 0 < position[2] < .04:
                continue
            if arm['gripper']['position'] >= .3 or i == 0 or rows[i-1]['arms']['left']['gripper']['position'] < .3:
                continue
            earlier = [r for r in rows if t-3 <= r['episode_time_s'] <= t]
            if max(r['arms']['left']['gripper']['position'] for r in earlier) < .6:
                continue
            # Locate descent before the hand pauses and closes around the pin.
            above = [r for r in earlier if r['arms']['left']['tool_pose_base']['position_m'][2] >= position[2] + .05]
            if not above:
                continue
            start = above[-1]
            displacement = np.array(position) - start['arms']['left']['tool_pose_base']['position_m']
            if displacement[2] > -.025 or np.linalg.norm(displacement) > .15:
                continue
            sample = dict(episode_id=entry['id'], time_s=t,
                          approach_start_time_s=start['episode_time_s'],
                          approach_direction_base=(displacement / np.linalg.norm(displacement)).tolist(),
                          approach_displacement_base_m=displacement.tolist())
            for side in ('left', 'right'):
                state = row['arms'][side]
                sample[side] = dict(joints_rad=state['position_rad'],
                                    gripper_rad=state['gripper']['position'],
                                    position_base_m=state['tool_pose_base']['position_m'],
                                    rotation_base=state['tool_pose_base']['rotation'])
            samples.append(sample)
            break
        if len(samples) >= limit:
            break
    if not samples:
        raise ValueError('No pin pickup windows match the original setup selection')
    return dict(schema_version=1, recordings=str(recordings), annotations=str(annotations),
                selection='Later left open-to-closed event below 40 mm and forward of x=370 mm in the original October 5 setup; 50 mm preceding descent. First episode visually checked against all three camera streams.',
                frame_mapping='Positions and rotation matrices map tool_link into each arm base frame; metres and radians. Apply each configured simulation base transform. Tool +X is the forward grasp axis. RPY uses Rz(yaw) Ry(pitch) Rx(roll). Camera/base extrinsics were not calibrated.',
                visual_reference=dict(episode_id='20261005T150601-420f52fb', pickup_time_s=[17,20], lifted_time_s=22),
                samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('sim/actuator/pin_regrasp_profile.json'))
    args = parser.parse_args()
    profile = build_profile(args.recordings, args.annotations)
    args.output.write_text(json.dumps(profile, indent=2) + '\n')
    rotations = [np.array(s['left']['rotation_base']) for s in profile['samples']]
    rpy = np.rad2deg([[np.arctan2(r[2,1], r[2,2]), np.arcsin(-r[2,0]), np.arctan2(r[1,0], r[0,0])] for r in rotations])
    print(f'{len(rotations)} pickups; median left roll/pitch/yaw degrees: {np.median(rpy, axis=0).round(2)}')
    print('Median approach direction:', np.median([s['approach_direction_base'] for s in profile['samples']], axis=0).round(3))


if __name__ == '__main__':
    main()
