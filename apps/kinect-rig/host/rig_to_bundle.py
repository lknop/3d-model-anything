#!/usr/bin/env python3
"""Convert a rig session (or a TUM RGB-D sequence) into a house-capture/5 session bundle, so the
scan-to-threejs scripts (reconstruct_bundle.py, bundle_to_room.py, prepare_photos.py) can use it.

usage: rig_to_bundle.py <session> -o BUNDLE [--kinect-poses trajectory.txt] [--calibration rig.json]
                        [--depth-every 3] [--photo-every 1.0]

Photos: the phone's keyframes (ARCore poses, 1920x1080) when the session has them; otherwise the
sharpest Kinect RGB frame of every --photo-every seconds. Depth: every --depth-every-th Kinect frame,
posed by one of
  --calibration    phone pose x phone-to-Kinect transform (rig.json: phoneToKinect, kinectLatencyMs,
                   kinectIntrinsics). Everything shares ARCore's gravity-aligned world.
  --kinect-poses   a Kinect trajectory from kinect_fusion.py (Kinect-only mode). Its world is the
                   first Kinect camera, turned so +Y is that camera's up: level only if it was.
                   Phone photos cannot be placed in it and are left out.
The bundle has no rooms yet (RoomPlan's role): room layout detection adds rooms/room-0.json.
Coordinates follow the bundle contract: meters, +Y up, camera looks down -Z, transforms are
column-major camera-to-world.
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import zlib

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sequence  # noqa: E402
import traj  # noqa: E402

CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])  # camera axes: x right/y down/z forward <-> x right/y up/z back
KINECT_EXPOSURE_S = 0.03  # nominal; the Kinect does not report its auto exposure


def column_major(m):
    return [float(v) for v in np.asarray(m).T.reshape(-1)]


def intrinsics3(fx, fy, cx, cy):
    return [fx, 0, 0, 0, fy, 0, cx, cy, 1]


def upright(cam_to_world):
    """How to turn the raw image upright, from where world +Y points in the camera (GL axes)."""
    up = cam_to_world[:3, :3].T @ np.array([0, 1.0, 0])
    options = {'none': up[1], 'cw90': -up[0], 'ccw90': up[0], '180': -up[1]}
    return max(options, key=options.get)


def phone_photos(rec_dir, out):
    """ARCore keyframes → photo records; images are copied unchanged (sensor orientation)."""
    records = []
    for line in open(os.path.join(rec_dir, 'keyframes.jsonl')):
        k = json.loads(line)
        src = os.path.join(rec_dir, k['file'])
        if not os.path.exists(src):
            continue
        m = np.eye(4)
        m[:3, :3] = Rotation.from_quat(k['pose'][3:7]).as_matrix()
        m[:3, 3] = k['pose'][:3]
        name = f'photos/{len(records):04d}.jpg'
        shutil.copyfile(src, os.path.join(out, name))
        i = k['intrinsics']
        exposure = k.get('exposure_ns')
        records.append({
            'file': name, 'room': 0, 'timestamp': k['t_ns'] / 1e9, 'width': i['width'], 'height': i['height'],
            'transform': column_major(m), 'intrinsics': intrinsics3(i['fx'], i['fy'], i['cx'], i['cy']),
            'tracking': 'normal', 'manual': False, 'exposure': exposure / 1e9 if exposure else None,
            'blurPixels': k.get('blur_px'), 'pass': 'detail', 'highRes': False, 'upright': upright(m),
            'camera': {'iso': k.get('iso'), 'shutter': exposure / 1e9 if exposure else None},
        })
    return records


def kinect_photos(frames, poses, intr, out, every_s):
    """Kinect-only mode: the sharpest RGB frame per time window becomes a photo."""
    records, window, best = [], None, None

    def flush():
        if best is None:
            return
        _, t, color_path, pose, blur = best
        name = f'photos/{len(records):04d}.jpg'
        shutil.copyfile(color_path, os.path.join(out, name))
        records.append({
            'file': name, 'room': 0, 'timestamp': t, 'width': 640, 'height': 480, 'transform': column_major(pose),
            'intrinsics': intrinsics3(*intr), 'tracking': 'normal', 'manual': False, 'exposure': None,
            'blurPixels': blur, 'pass': 'detail', 'highRes': False, 'upright': upright(pose), 'camera': None,
        })

    prev = None
    for (t, _, color_path), pose in zip(frames, poses):
        if pose is None:
            continue
        blur = None
        if prev is not None and t > prev[0]:
            angle = np.linalg.norm(Rotation.from_matrix(prev[1][:3, :3].T @ pose[:3, :3]).as_rotvec())
            blur = round(float(angle / (t - prev[0]) * intr[0] * KINECT_EXPOSURE_S), 2)
        prev = (t, pose)
        if window is None or t - window >= every_s:
            flush()
            window, best = t, None
        gray = cv2.imread(color_path, cv2.IMREAD_GRAYSCALE)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        if best is None or sharp > best[0]:
            best = (sharp, t, color_path, pose, blur)
    flush()
    return records


def write_depth(frames, poses, intr, depth_scale, out, every):
    records = []
    for n, ((t, depth_path, color_path), pose) in enumerate(zip(frames, poses)):
        if pose is None or n % every:
            continue
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        meters = (depth.astype(np.float32) / depth_scale).astype('<f2')
        stem = f'depth/{len(records):05d}'
        packer = zlib.compressobj(6, zlib.DEFLATED, -15)  # raw DEFLATE, as the app writes
        with open(os.path.join(out, stem + '.f16z'), 'wb') as f:
            f.write(packer.compress(meters.tobytes()) + packer.flush())
        np.where(depth > 0, 2, 0).astype(np.uint8).tofile(os.path.join(out, stem + '.conf'))  # no confidence: valid = high
        shutil.copyfile(color_path, os.path.join(out, stem + '.jpg'))
        h, w = depth.shape
        records.append({'file': stem + '.f16z', 'confidence': stem + '.conf', 'rgb': stem + '.jpg',
                        'rgbIntrinsics': intrinsics3(*intr), 'room': 0, 'timestamp': t, 'width': w, 'height': h,
                        'transform': column_major(pose), 'intrinsics': intrinsics3(*intr)})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', help='rig session folder or TUM RGB-D sequence')
    ap.add_argument('-o', required=True, help='new bundle folder')
    ap.add_argument('--kinect-poses', help='Kinect-only mode: trajectory.txt from kinect_fusion.py')
    ap.add_argument('--calibration', help='rig mode: rig.json with phoneToKinect, kinectLatencyMs, kinectIntrinsics')
    ap.add_argument('--depth-every', type=int, default=3, help='keep every Nth Kinect frame as a depth frame')
    ap.add_argument('--photo-every', type=float, default=1.0, help='Kinect-only photos: one per this many seconds')
    a = ap.parse_args()
    if os.path.exists(a.o):
        ap.error(f'{a.o} exists; choose a new folder')
    if a.kinect_poses and a.calibration:
        ap.error('use --kinect-poses or --calibration, not both')

    is_tum = os.path.exists(os.path.join(a.session, 'depth.txt'))
    rec_dir = os.path.join(a.session, 'phone', 'recording')
    has_phone = os.path.exists(os.path.join(rec_dir, 'keyframes.jsonl'))
    has_kinect = is_tum or os.path.exists(os.path.join(a.session, 'kinect', 'frames.jsonl'))
    os.makedirs(os.path.join(a.o, 'photos'))
    os.makedirs(os.path.join(a.o, 'depth'))
    notes = []

    frames, depth_scale, intr = sequence.load(a.session) if has_kinect else ([], 1000.0, sequence.DEFAULT_INTRINSICS)
    kinect_poses = [None] * len(frames)
    if has_kinect and a.kinect_poses:
        times, poses = traj.read_tum(a.kinect_poses)
        to_gl_world = CV_TO_GL  # first camera's y-down/z-forward world → y-up/z-back
        kinect_poses = [None if p is None else to_gl_world @ p @ CV_TO_GL
                        for p in (traj.interpolate(times, poses, t) for t, _, _ in frames)]
        notes.append('Kinect-only: world = first Kinect camera with +Y as its up, not leveled to gravity')
    elif has_kinect and a.calibration:
        cal = json.load(open(a.calibration))
        intr = tuple(cal.get('kinectIntrinsics', intr))
        phone_to_kinect = np.array(cal['phoneToKinect'], dtype=np.float64).reshape(4, 4).T
        latency = cal.get('kinectLatencyMs', 0) / 1e3
        session = json.load(open(os.path.join(a.session, 'session.json')))
        offset = session['clock']['offset_ns']  # phone clock - laptop clock
        rows = [json.loads(line) for line in open(os.path.join(rec_dir, 'poses.jsonl'))]
        rows = [r for r in rows if r['tracking'] == 'TRACKING']
        p_times = np.array([(r['t_ns'] - offset) / 1e9 for r in rows])  # phone camera time on the laptop clock
        p_poses = np.tile(np.eye(4), (len(rows), 1, 1))
        p_poses[:, :3, :3] = Rotation.from_quat([r['pose'][3:7] for r in rows]).as_matrix()
        p_poses[:, :3, 3] = [r['pose'][:3] for r in rows]
        kinect_poses = []
        for t, _, _ in frames:
            phone = traj.interpolate(p_times, p_poses, t - latency)
            kinect_poses.append(None if phone is None else phone @ phone_to_kinect)
        notes.append(f'rig: Kinect posed from ARCore via {os.path.basename(a.calibration)}')
    elif has_kinect:
        notes.append('Kinect frames skipped: give --kinect-poses (Kinect-only) or --calibration (rig)')

    if has_phone and not a.kinect_poses:
        photos = phone_photos(rec_dir, a.o)
    elif has_kinect and a.kinect_poses:
        photos = kinect_photos(frames, kinect_poses, intr, a.o, a.photo_every)
        if has_phone:
            notes.append('phone keyframes left out: they live in the ARCore world, not the Kinect-only one')
    else:
        photos = []
    depth = write_depth(frames, kinect_poses, intr, depth_scale, a.o, a.depth_every) if any(p is not None for p in kinect_poses) else []

    json.dump(photos, open(os.path.join(a.o, 'photos/photos.json'), 'w'), indent=1)
    json.dump(depth, open(os.path.join(a.o, 'depth/frames.json'), 'w'), indent=1)
    name = os.path.basename(os.path.normpath(a.session))
    manifest = {
        'format': 'house-capture/5',
        'createdAt': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'device': 'Kinect for Windows v1 + Android ARCore rig' if not is_tum else 'TUM RGB-D (Kinect v1)',
        'systemVersion': 'kinect-rig',
        'coordinateSystem': 'Meters, +Y up, right-handed. Rig mode: ARCore world (gravity-aligned). '
                            'Kinect-only mode: first Kinect camera, +Y = its up. Photos and depth frames share it.',
        'house': 'kinect-rig', 'session': name, 'zone': 'default',
        'rooms': [], 'photos': 'photos/photos.json', 'photoCount': len(photos),
        'depthFrames': 'depth/frames.json', 'depthFrameCount': len(depth), 'mesh': None,
        'kinectRig': {'source': os.path.abspath(a.session), 'notes': notes,
                      'kinectIntrinsics': list(intr), 'calibrated': bool(a.calibration)},
    }
    json.dump(manifest, open(os.path.join(a.o, 'manifest.json'), 'w'), indent=2)
    print(f'{len(photos)} photos, {len(depth)} depth frames → {a.o}')
    for n in notes:
        print(f'  note: {n}')


if __name__ == '__main__':
    main()
