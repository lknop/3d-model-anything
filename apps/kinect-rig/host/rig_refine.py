#!/usr/bin/env python3
"""Undo ARCore's map re-anchoring jumps by aligning the Kinect depth of each stretch between them.

usage: rig_refine.py <session> --prior prior.txt [--calibration rig.json] -o refined.txt [--report report.json]

ARCore is consistent between its jumps (it occasionally re-anchors its map and shifts every later
pose by ~5-15 cm at once, never revisiting earlier ones). So: split the prior (rig_to_bundle.py
--write-prior) at unphysical jumps, build each segment's point cloud from its Kinect depth with
ARCore's poses, and align every segment rigidly (point-to-plane ICP) to all earlier ones. Each
segment gets one correction, so the result cannot drift the way frame-by-frame tracking does.

Writes the refined Kinect trajectory (TUM, CV camera axes, ARCore world) for rig_to_bundle.py --refined.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sequence  # noqa: E402
import traj  # noqa: E402

JUMP_SPEED = 1.5  # m/s between consecutive frames: faster is a re-anchoring jump, not motion
JUMP_TURN = np.radians(180)  # rad/s
MIN_SEGMENT_FRAMES = 15  # shorter stretches keep their neighbor's correction
DEPTH_RANGE = (0.5, 3.0)  # m; Kinect v1 depth is coarse beyond ~3 m
PIXEL_STEP = 4  # use every 4th pixel per axis
MIN_FITNESS = 0.3  # fraction of a segment's points that must overlap the reference


def split(times, poses):
    """Start index of every segment: the first frame and every frame after a jump."""
    starts = [0]
    for i in range(1, len(times)):
        delta = np.linalg.inv(poses[i - 1]) @ poses[i]
        dt = max(times[i] - times[i - 1], 1e-3)
        turn = np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
        if np.linalg.norm(delta[:3, 3]) > JUMP_SPEED * dt or turn > JUMP_TURN * dt:
            starts.append(i)
    # fold very short segments into the previous one
    merged = [starts[0]]
    for s, e in zip(starts[1:], starts[2:] + [len(times)]):
        if e - s >= MIN_SEGMENT_FRAMES:
            merged.append(s)
    return merged


def cloud(depth_paths, poses, intr, depth_scale, voxel):
    """World-space point cloud from depth frames (camera-to-world poses, CV axes)."""
    fx, fy, cx, cy = intr
    pts = []
    for path, pose in zip(depth_paths, poses):
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)[::PIXEL_STEP, ::PIXEL_STEP].astype(np.float32) / depth_scale
        v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]] * PIXEL_STEP
        ok = (d > DEPTH_RANGE[0]) & (d < DEPTH_RANGE[1])
        z = d[ok]
        cam = np.c_[(u[ok] - cx) * z / fx, (v[ok] - cy) * z / fy, z]
        pts.append(cam @ pose[:3, :3].T + pose[:3, 3])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts) if pts else np.zeros((0, 3))))
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    return pc


def align(source, target):
    """Coarse-to-fine point-to-plane ICP; returns (transform, fitness, rmse) at the finest level."""
    reg = o3d.pipelines.registration
    t = np.eye(4)
    result = None
    for dist, iters in ((0.25, 30), (0.10, 30), (0.04, 50)):
        loss = reg.TukeyLoss(k=dist)
        result = reg.registration_icp(source, target, dist, t, reg.TransformationEstimationPointToPlane(loss),
                                      reg.ICPConvergenceCriteria(max_iteration=iters))
        t = result.transformation
    return t, result.fitness, result.inlier_rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--prior', required=True, help='rig_to_bundle.py --write-prior output')
    ap.add_argument('--calibration', help='rig.json, for the calibrated Kinect intrinsics')
    ap.add_argument('-o', required=True, help='refined trajectory (TUM)')
    ap.add_argument('--report', help='write per-segment details as JSON')
    ap.add_argument('--every', type=int, default=3, help='use every Nth Kinect frame for the clouds')
    ap.add_argument('--voxel', type=float, default=0.02)
    a = ap.parse_args()

    frames, depth_scale, intr = sequence.load(a.session)
    if a.calibration:
        intr = tuple(json.load(open(a.calibration))['kinectIntrinsics'])
    times, poses = traj.read_tum(a.prior)
    by_time = {round(t, 6): f[1] for f in frames for t in [f[0]]}
    depth_paths = [by_time.get(round(t, 6)) for t in times]
    starts = split(times, poses)
    bounds = list(zip(starts, starts[1:] + [len(times)]))
    print(f'{len(times)} prior poses → {len(bounds)} segments between ARCore jumps')

    corrections = [np.eye(4)] * len(bounds)
    reference = None
    report = []
    for n, (s, e) in enumerate(bounds):
        idx = [i for i in range(s, e, a.every) if depth_paths[i]]
        seg = cloud([depth_paths[i] for i in idx], poses[idx], intr, depth_scale, a.voxel)
        entry = {'segment': n, 'start_s': round(times[s] - times[0], 2), 'end_s': round(times[e - 1] - times[0], 2),
                 'frames': e - s, 'points': len(seg.points)}
        if reference is None:
            if len(seg.points) > 1000:
                reference = seg
                entry['role'] = 'reference'
        else:
            t, fitness, rmse = align(seg, reference)
            move = np.linalg.norm(t[:3, 3])
            turn = np.degrees(np.linalg.norm(Rotation.from_matrix(t[:3, :3]).as_rotvec()))
            entry.update(fitness=round(fitness, 3), rmse_cm=round(rmse * 100, 2), move_cm=round(move * 100, 1), turn_deg=round(turn, 2))
            if fitness >= MIN_FITNESS and move < 0.5 and turn < 10:
                corrections[n] = t
                seg.transform(t)
                entry['role'] = 'aligned'
            else:
                corrections[n] = corrections[n - 1]  # too little overlap: carry the previous correction
                seg.transform(corrections[n])
                entry['role'] = 'kept previous correction'
            reference += seg
            reference = reference.voxel_down_sample(a.voxel)
            reference.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=a.voxel * 4, max_nn=30))
        report.append(entry)
        extra = '' if 'fitness' not in entry else (f": moved {entry['move_cm']} cm, turned {entry['turn_deg']} deg, "
                                                     f"overlap {entry['fitness']:.0%}, residual {entry['rmse_cm']} cm")
        print(f"  segment {n}: {entry['start_s']:6.1f}-{entry['end_s']:6.1f} s, {entry['frames']} frames, "
              f"{entry['points']} points, {entry.get('role', 'too small, skipped')}{extra}")

    refined = np.array([corrections[n] @ poses[i] for n, (s, e) in enumerate(bounds) for i in range(s, e)])
    traj.write_tum(a.o, times, refined, 'Kinect camera-to-world refined per ARCore segment; CV camera axes, ARCore world (y up)')
    if a.report:
        with open(a.report, 'w') as f:
            json.dump({'segments': report, 'jumpSpeed': JUMP_SPEED, 'voxel': a.voxel}, f, indent=2)
    print(f'→ {a.o}')


if __name__ == '__main__':
    main()
