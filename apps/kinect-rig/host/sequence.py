"""Kinect RGB-D sequences: TUM RGB-D benchmark folders and rig sessions, as one frame list.

load(path) -> (frames, depth_scale, intrinsics) with frames = [(time_s, depth_path, color_path)],
depth stored as uint16 (depth_scale units per meter) and registered to the color image.
"""
import json
import os

import numpy as np

import traj

# Registered Kinect v1 depth uses the RGB camera's intrinsics. TUM recommends the ROS defaults
# for freiburg1/2 (their depth is pre-registered with them) and publishes calibrated freiburg3 values.
DEFAULT_INTRINSICS = (525.0, 525.0, 319.5, 239.5)
TUM_FREIBURG3 = (535.4, 539.2, 320.1, 247.6)


def tum_sequence(path):
    def listing(name):
        rows = [line.split() for line in open(os.path.join(path, name)) if line.strip() and not line.startswith('#')]
        return np.array([float(r[0]) for r in rows]), [os.path.join(path, r[1]) for r in rows]
    dt, dfiles = listing('depth.txt')
    ct, cfiles = listing('rgb.txt')
    frames = [(dt[i], dfiles[i], cfiles[j]) for i, j in traj.associate(dt, ct, 0.02)]
    frames.sort()
    intr = TUM_FREIBURG3 if 'freiburg3' in os.path.basename(os.path.normpath(path)) else DEFAULT_INTRINSICS
    return frames, 5000.0, intr


def rig_sequence(path):
    rows = [json.loads(line) for line in open(os.path.join(path, 'kinect', 'frames.jsonl'))]
    depth = [r for r in rows if r['stream'] == 'depth']
    rgb = [r for r in rows if r['stream'] == 'rgb']
    dt = np.array([r['laptop_ns'] / 1e9 for r in depth])
    ct = np.array([r['laptop_ns'] / 1e9 for r in rgb])
    base = os.path.join(path, 'kinect')
    frames = [(dt[i], os.path.join(base, depth[i]['file']), os.path.join(base, rgb[j]['file']))
              for i, j in traj.associate(dt, ct, 0.02)]
    frames.sort()
    return frames, 1000.0, DEFAULT_INTRINSICS


def load(path):
    return (tum_sequence if os.path.exists(os.path.join(path, 'depth.txt')) else rig_sequence)(path)
