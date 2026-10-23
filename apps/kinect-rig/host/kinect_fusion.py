#!/usr/bin/env python3
"""Track and fuse a Kinect RGB-D sequence on the GPU (Open3D dense SLAM, KinectFusion-style).

usage: kinect_fusion.py <sequence> -o OUT [--poses traj.txt] [--gt groundtruth.txt] [--every N]
                        [--voxel 0.01] [--depth-max 3.0] [--intrinsics fx fy cx cy] [--device CUDA:0]

<sequence> is a TUM RGB-D folder (rgb.txt, depth.txt) or a rig session (kinect/frames.jsonl).
Without --poses every frame is tracked against a raycast of the model fused so far (Kinect-only
mode; the world frame is the first camera). With --poses (TUM format, camera-to-world, CV axes)
those poses are used as given and only fused. --gt reports the absolute trajectory error.

Writes OUT/trajectory.txt (TUM format, camera-to-world, x right / y down / z forward),
OUT/mesh.ply (vertex colors) and OUT/report.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import open3d as o3d
import open3d.core as o3c

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sequence  # noqa: E402
import traj  # noqa: E402

LOST_FITNESS = 0.3  # fraction of pixels with a model correspondence below which a frame is "lost"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sequence')
    ap.add_argument('-o', required=True)
    ap.add_argument('--poses', help='camera-to-world poses (TUM format); fuse only, no tracking')
    ap.add_argument('--gt', help='ground-truth trajectory (TUM format) for the ATE report')
    ap.add_argument('--every', type=int, default=1, help='use every Nth frame')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--voxel', type=float, default=0.01)
    ap.add_argument('--depth-max', type=float, default=3.0)
    ap.add_argument('--trunc', type=float, default=8.0, help='TSDF truncation in voxels')
    ap.add_argument('--block-count', type=int, default=40000)
    ap.add_argument('--intrinsics', type=float, nargs=4, metavar=('FX', 'FY', 'CX', 'CY'))
    ap.add_argument('--device', default='CUDA:0')
    a = ap.parse_args()

    frames, depth_scale, intr = sequence.load(a.sequence)
    if a.intrinsics:
        intr = tuple(a.intrinsics)
    frames = frames[::a.every]
    if a.max_frames:
        frames = frames[:a.max_frames]
    given = traj.read_tum(a.poses) if a.poses else None
    os.makedirs(a.o, exist_ok=True)

    device = o3c.Device(a.device)
    fx, fy, cx, cy = intr
    k = o3c.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3c.Dtype.Float64)
    first = o3d.t.io.read_image(frames[0][1])
    h, w = first.rows, first.columns
    pose = np.eye(4)
    if given is not None:
        start = traj.interpolate(*given, frames[0][0])
        if start is None:
            ap.error('--poses does not cover the first frame')
        pose = start
    model = o3d.t.pipelines.slam.Model(a.voxel, 16, a.block_count, o3c.Tensor(pose), device)
    input_frame = o3d.t.pipelines.slam.Frame(h, w, k, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(h, w, k, device)

    times, poses, fitness, lost = [], [], [], 0
    t0 = time.time()
    for i, (t, depth_path, color_path) in enumerate(frames):
        depth = o3d.t.io.read_image(depth_path).to(device)
        color = o3d.t.io.read_image(color_path).to(device)
        input_frame.set_data_from_image('depth', depth)
        input_frame.set_data_from_image('color', color)
        if given is not None:
            p = traj.interpolate(*given, t)
            if p is None:
                continue
            pose = p
        elif i > 0:
            try:
                result = model.track_frame_to_model(input_frame, raycast_frame, depth_scale, a.depth_max)
                fit = float(result.fitness)
            except RuntimeError:  # "Singular 6x6 linear system": too little geometry to align
                fit = 0.0
            fitness.append(fit)
            if fit < LOST_FITNESS:
                # Keep the last pose and don't fuse: a bad pose would corrupt the model,
                # and the next frames may still find it again.
                lost += 1
                times.append(t)
                poses.append(pose.copy())
                continue
            pose = pose @ result.transformation.cpu().numpy()
        times.append(t)
        poses.append(pose.copy())
        model.update_frame_pose(i, o3c.Tensor(pose))
        model.integrate(input_frame, depth_scale, a.depth_max, a.trunc)
        model.synthesize_model_frame(raycast_frame, depth_scale, 0.1, a.depth_max, a.trunc, False)
        if i % 100 == 0:
            print(f'\r{i}/{len(frames)} frames, {i / max(time.time() - t0, 1e-9):.1f} fps, lost {lost}', end='', flush=True)
    seconds = time.time() - t0
    print(f'\r{len(times)} frames fused in {seconds:.0f} s ({len(times) / seconds:.1f} fps), lost {lost}')

    times, poses = np.array(times), np.array(poses)
    traj.write_tum(os.path.join(a.o, 'trajectory.txt'), times, poses, 'camera-to-world, x right, y down, z forward')
    mesh = model.extract_trianglemesh(weight_threshold=3.0).to_legacy()
    o3d.io.write_triangle_mesh(os.path.join(a.o, 'mesh.ply'), mesh)
    report = {'sequence': os.path.abspath(a.sequence), 'frames': len(times), 'seconds': round(seconds, 1),
              'fps': round(len(times) / seconds, 1), 'mode': 'given poses' if given is not None else 'frame-to-model tracking',
              'lostFrames': lost, 'intrinsics': intr, 'voxel': a.voxel, 'depthMax': a.depth_max, 'every': a.every,
              'triangles': len(mesh.triangles)}
    if fitness:
        report['fitness'] = {'min': min(fitness), 'median': float(np.median(fitness))}
    if a.gt:
        report['ate'] = traj.ate(times, poses, *traj.read_tum(a.gt))
        e = report['ate']
        print(f"ATE rmse {e['rmse'] * 100:.1f} cm, median {e['median'] * 100:.1f} cm, max {e['max'] * 100:.1f} cm "
              f"over {e['gtLength']:.1f} m of ground-truth path ({e['pairs']} poses)")
    with open(os.path.join(a.o, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'{len(mesh.triangles):,} triangles → {a.o}')


if __name__ == '__main__':
    main()
