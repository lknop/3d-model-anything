"""Camera trajectories: TUM text format, timestamp association, rigid alignment and ATE.

A trajectory is (times [N] seconds, poses [N,4,4] camera-to-world). TUM lines are
"timestamp tx ty tz qx qy qz qw" with '#' comments.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def read_tum(path):
    rows = [line.split() for line in open(path) if line.strip() and not line.startswith('#')]
    data = np.array(rows, dtype=np.float64)
    poses = np.tile(np.eye(4), (len(data), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(data[:, 4:8]).as_matrix()
    poses[:, :3, 3] = data[:, 1:4]
    return data[:, 0], poses


def write_tum(path, times, poses, header=None):
    q = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    with open(path, 'w') as f:
        if header:
            f.write(f'# {header}\n')
        for t, p, qq in zip(times, poses, q):
            f.write(f'{t:.6f} {p[0, 3]:.6f} {p[1, 3]:.6f} {p[2, 3]:.6f} {qq[0]:.7f} {qq[1]:.7f} {qq[2]:.7f} {qq[3]:.7f}\n')


def associate(a, b, max_diff=0.02):
    """Index pairs (i, j) matching times a[i] ~ b[j] one-to-one, nearest first."""
    b_sorted = np.argsort(b)
    bs = b[b_sorted]
    pairs, used = [], set()
    for i, t in enumerate(a):
        k = np.searchsorted(bs, t)
        best = min((c for c in (k - 1, k) if 0 <= c < len(bs)), key=lambda c: abs(bs[c] - t), default=None)
        if best is not None and abs(bs[best] - t) <= max_diff and best not in used:
            used.add(best)
            pairs.append((i, int(b_sorted[best])))
    return pairs


def align_rigid(src, dst):
    """Rotation R and translation t minimizing |R src + t - dst| (Horn/Umeyama, no scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    s = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ s @ vt
    return r, mu_d - r @ mu_s


def ate(est_times, est_poses, gt_times, gt_poses, max_diff=0.02):
    """Absolute trajectory error after rigid alignment: rmse/mean/median/max in meters."""
    pairs = associate(est_times, gt_times, max_diff)
    if len(pairs) < 3:
        raise ValueError('fewer than 3 timestamps match the ground truth')
    est = np.array([est_poses[i][:3, 3] for i, _ in pairs])
    gt = np.array([gt_poses[j][:3, 3] for _, j in pairs])
    r, t = align_rigid(est, gt)
    err = np.linalg.norm((r @ est.T).T + t - gt, axis=1)
    return {'pairs': len(pairs), 'rmse': float(np.sqrt((err ** 2).mean())), 'mean': float(err.mean()),
            'median': float(np.median(err)), 'max': float(err.max()),
            'gtLength': float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())}


def interpolate(times, poses, t):
    """Pose at time t: linear translation, spherical rotation; None outside the range."""
    if t < times[0] or t > times[-1]:
        return None
    k = int(np.searchsorted(times, t))
    if k == 0 or times[k] == t:
        return poses[k].copy()
    t0, t1 = times[k - 1], times[k]
    a = (t - t0) / (t1 - t0)
    out = np.eye(4)
    out[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([poses[k - 1][:3, :3], poses[k][:3, :3]])))(a).as_matrix()
    out[:3, 3] = (1 - a) * poses[k - 1][:3, 3] + a * poses[k][:3, 3]
    return out
