#!/usr/bin/env python3
"""Calibrate a rig from a recorded session: Kinect RGB intrinsics, the phone-to-Kinect transform
and the Kinect's delay relative to the phone.

usage: rig_calibrate.py <session> -o rig.json [--square 0.0254] [--pattern 9 6]

Record the session with rig_record.py while pointing the rig at a printed checkerboard from ~15
viewpoints, holding still at each until the phone beeps, then turn the rig around for 10-15 s.

  1. delay: the Kinect's image motion (optical flow) against ARCore's angular speed, best lag
  2. Kinect intrinsics: pinhole calibration on Kinect frames that see the board while still
  3. transform: per phone keyframe that sees the board, solvePnP in both cameras;
     phoneToKinect = phone_T_board * inverse(kinect_T_board), averaged with outliers dropped

Writes rig.json for rig_to_bundle.py --calibration: phoneToKinect is the Kinect RGB camera's pose
in the phone camera's frame, column-major 4x4, both in the bundle's camera axes (x right, y up,
z back, sensor orientation); kinectLatencyMs is how much later a Kinect frame reaches the laptop
than its exposure, relative to the phone clock mapping.
"""
import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
MAX_LAG_S = 0.25


def load(session):
    s = json.load(open(os.path.join(session, 'session.json')))
    if not s.get('clock'):
        raise SystemExit('session has no phone clock samples: record with the phone connected')
    offset = s['clock']['offset_ns']
    rec = os.path.join(session, 'phone', 'recording')
    poses = [json.loads(line) for line in open(os.path.join(rec, 'poses.jsonl'))]
    poses = [p for p in poses if p['tracking'] == 'TRACKING']
    keyframes = [json.loads(line) for line in open(os.path.join(rec, 'keyframes.jsonl'))]
    for k in keyframes:
        k['laptop_s'] = (k['t_ns'] - offset) / 1e9
        k['path'] = os.path.join(rec, k['file'])
    p_times = np.array([(p['t_ns'] - offset) / 1e9 for p in poses])
    p_rot = Rotation.from_quat([p['pose'][3:7] for p in poses])
    rows = [json.loads(line) for line in open(os.path.join(session, 'kinect', 'frames.jsonl'))]
    rgb = [r for r in rows if r['stream'] == 'rgb']
    k_times = np.array([r['laptop_ns'] / 1e9 for r in rgb])
    k_paths = [os.path.join(session, 'kinect', r['file']) for r in rgb]
    return keyframes, p_times, p_rot, k_times, k_paths


def estimate_delay(p_times, p_rot, k_times, k_paths):
    """Lag (s) that best lines up Kinect image motion with phone angular speed."""
    flow, prev = [], None
    for path in k_paths:
        g = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (160, 120)).astype(np.float32)
        if prev is None:
            flow.append(0.0)
        else:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g)
            flow.append(float(np.hypot(dx, dy)))
        prev = g
    flow = np.array(flow) / np.gradient(k_times).clip(1e-3)
    rel = (p_rot[:-1].inv() * p_rot[1:]).magnitude() / np.diff(p_times).clip(1e-3)
    speed_t = (p_times[:-1] + p_times[1:]) / 2
    lags = np.arange(-MAX_LAG_S, MAX_LAG_S + 1e-9, 0.005)
    scores = []
    for lag in lags:
        phone_at_kinect = np.interp(k_times - lag, speed_t, rel, left=np.nan, right=np.nan)
        ok = np.isfinite(phone_at_kinect)
        scores.append(np.corrcoef(phone_at_kinect[ok], flow[ok])[0, 1] if ok.sum() > 30 else -1)
    best = int(np.nanargmax(scores))
    return float(lags[best]), float(scores[best])


def board_points(pattern, square):
    cols, rows = pattern
    grid = np.zeros((cols * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return grid


def find_board(path, pattern):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    return (corners.reshape(-1, 2).astype(np.float32), gray.shape[::-1]) if ok else (None, gray.shape[::-1])


def pose_from(obj, img, k):
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    m = np.eye(4)
    m[:3, :3] = cv2.Rodrigues(rvec)[0]
    m[:3, 3] = tvec.ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
    return m, float(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('-o', required=True)
    ap.add_argument('--square', type=float, default=0.0254, help='checkerboard square size in meters (measure it)')
    ap.add_argument('--pattern', type=int, nargs=2, default=(9, 6), help='inner corners (columns rows)')
    a = ap.parse_args()
    pattern = tuple(a.pattern)
    obj = board_points(pattern, a.square)
    keyframes, p_times, p_rot, k_times, k_paths = load(a.session)

    delay, corr = estimate_delay(p_times, p_rot, k_times, k_paths)
    print(f'Kinect delay {delay * 1000:+.0f} ms (motion correlation {corr:.2f})')

    # pair each still phone keyframe with the Kinect frame exposed at the same moment
    pairs = []
    for kf in keyframes:
        j = int(np.argmin(np.abs(k_times - (kf['laptop_s'] + delay))))
        if abs(k_times[j] - (kf['laptop_s'] + delay)) > 0.05:
            continue
        phone_corners, _ = find_board(kf['path'], pattern)
        kinect_corners, kinect_size = find_board(k_paths[j], pattern)
        if phone_corners is not None and kinect_corners is not None:
            pairs.append((kf, phone_corners, kinect_corners))
    print(f'{len(keyframes)} phone keyframes, {len(pairs)} with the board seen by both cameras')
    if len(pairs) < 5:
        raise SystemExit('need at least 5 views of the board in both cameras: record more viewpoints, closer')

    # Kinect pinhole intrinsics (the pipeline is pinhole); distortion fitted separately for information
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    rms, k_kinect, _, _, _ = cv2.calibrateCamera([obj] * len(pairs), [p[2] for p in pairs], kinect_size, None, None, flags=flags)
    rms_d, _, dist, _, _ = cv2.calibrateCamera([obj] * len(pairs), [p[2] for p in pairs], kinect_size, None, None)
    print(f'Kinect intrinsics fx {k_kinect[0, 0]:.1f} fy {k_kinect[1, 1]:.1f} cx {k_kinect[0, 2]:.1f} cy {k_kinect[1, 2]:.1f}, '
          f'reprojection {rms:.2f} px (with distortion {rms_d:.2f} px, k1 {dist[0, 0]:+.3f})')

    estimates = []
    for kf, phone_corners, kinect_corners in pairs:
        i = kf['intrinsics']
        k_phone = np.array([[i['fx'], 0, i['cx']], [0, i['fy'], i['cy']], [0, 0, 1]])
        phone_T_board, e_phone = pose_from(obj, phone_corners, k_phone)
        kinect_T_board, e_kinect = pose_from(obj, kinect_corners, k_kinect)
        if phone_T_board is None or kinect_T_board is None:
            continue
        estimates.append((phone_T_board @ np.linalg.inv(kinect_T_board), e_phone, e_kinect, kf['i']))
    rots = Rotation.from_matrix([e[0][:3, :3] for e in estimates])
    trans = np.array([e[0][:3, 3] for e in estimates])
    mean_rot, mean_t = rots.mean(), np.median(trans, axis=0)
    dev_deg = np.degrees((rots * mean_rot.inv()).magnitude())
    dev_mm = np.linalg.norm(trans - mean_t, axis=1) * 1000
    keep = (dev_deg <= max(3 * np.median(dev_deg), 0.5)) & (dev_mm <= max(3 * np.median(dev_mm), 10))
    rots, trans = Rotation.from_matrix(rots.as_matrix()[keep]), trans[keep]
    mean_rot, mean_t = rots.mean(), trans.mean(axis=0)
    dev_deg = np.degrees((rots * mean_rot.inv()).magnitude())
    dev_mm = np.linalg.norm(trans - mean_t, axis=1) * 1000

    cv_pose = np.eye(4)
    cv_pose[:3, :3] = mean_rot.as_matrix()
    cv_pose[:3, 3] = mean_t
    gl_pose = CV_TO_GL @ cv_pose @ CV_TO_GL  # same axis flip on both cameras
    euler = Rotation.from_matrix(gl_pose[:3, :3]).as_euler('xyz', degrees=True)
    print(f'phone → Kinect: offset {np.round(gl_pose[:3, 3] * 100, 1)} cm, rotation xyz {np.round(euler, 1)} deg '
          f'(from {keep.sum()}/{len(keep)} views; spread {np.median(dev_deg):.2f} deg, {np.median(dev_mm):.1f} mm median)')

    result = {
        'format': 'kinect-rig-calibration/1',
        'createdAt': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'session': os.path.abspath(a.session),
        'phoneToKinect': [float(v) for v in gl_pose.T.reshape(-1)],
        'kinectLatencyMs': round(delay * 1000, 1),
        'kinectIntrinsics': [float(k_kinect[0, 0]), float(k_kinect[1, 1]), float(k_kinect[0, 2]), float(k_kinect[1, 2])],
        'kinectDistortion': [float(v) for v in dist.ravel()],
        'quality': {'views': int(keep.sum()), 'viewsRejected': int((~keep).sum()), 'motionCorrelation': round(corr, 3),
                    'kinectReprojectionPx': round(float(rms), 3),
                    'spreadDegMedian': round(float(np.median(dev_deg)), 3), 'spreadMmMedian': round(float(np.median(dev_mm)), 2)},
        'board': {'innerCorners': list(pattern), 'squareM': a.square},
        'conventions': 'phoneToKinect: Kinect RGB camera pose in the phone camera frame; camera axes x right, y up, '
                       'z back (sensor orientation); column-major. Kinect pose = phone pose @ phoneToKinect.',
    }
    with open(a.o, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'→ {a.o}')


if __name__ == '__main__':
    main()
