#!/usr/bin/env python3
"""Detect a room's layout in a fused scan and write it as RoomPlan-style JSON (rooms/room-0.json),
standing in for Apple's RoomPlan so bundle_to_room.py and the scan-to-threejs workflow can run.

usage: rig_layout.py <bundle> --mesh scan_world.ply [--name NAME] [--plan plan.png]

Works on the mesh from reconstruct_bundle.py (bundle world: meters, +Y up, gravity-aligned).
  floor            most common height of upward-facing surfaces
  ceiling faces    every large planar downward-facing surface, flat or sloped at any orientation
                   (attic roofs: slopes, hip triangles), each with its observed outline; edges where
                   two faces meet are classed peak (ridge, hip) or valley; flat ceiling = a flat
                   face of at least 1 m^2
  wall tops        follow the lowest ceiling face above them (eaves, knee walls, gable ends), stored
                   as a world-space profile; the wall's RoomPlan height is the profile's lowest point
  heading          dominant direction of vertical surfaces; walls are assumed to meet at right angles
  walls            peaks of vertical surface offsets along both directions that are tall and
                   outermost (little solid surface behind them), ends snapped into corners
  openings         wall gaps of 0.6-1.3 m (door) or wider (opening) where the scan sees through
                   to space beyond the wall; gaps with nothing behind them are just unscanned
  floor outline    floor coverage, closed over furniture, simplified
  objects          clusters of what is left, as heading-aligned boxes, category "object"
Windows are not detected yet. Every element carries a confidence; check the plan against photos.
Ceilings are only as complete as the scan: point the rig up at slopes and edges while scanning.
"""
import argparse
import datetime
import json
import math
import os
import uuid

import cv2
import numpy as np
import open3d as o3d

GRID = 0.05  # m, raster resolution for floor coverage and gap analysis
WALL_BAND = 0.04  # m, half-thickness of a wall plane's inliers
MIN_WALL_AREA = 0.5  # m^2 of observed surface
MIN_WALL_LENGTH = 0.4
MIN_WALL_TOP = 0.9  # m above the floor seen on the surface (scans held low see little more)
MAX_WALL_BOTTOM = 0.4  # m: walls reach the floor; table and shelf sides above it do not
BEHIND = 0.15  # m: solid surface this far beyond a candidate wall means it is not the room's boundary
SNAP = 0.35  # m: wall ends this close to a perpendicular wall meet it
MIN_FACE_AREA = 0.4  # m^2 of connected observed surface per ceiling face
MIN_FLAT_AREA = 1.0  # m^2: a flat face this large counts as a flat ceiling
FLAT_PITCH = 8.0  # deg
DOOR_WIDTH = (0.6, 1.3)
DOOR_HEIGHT = 2.05


def to_frame(theta):
    """World x/z → heading frame a/b (rotation about +Y by -theta) and back."""
    c, s = math.cos(theta), math.sin(theta)
    fwd = lambda x, z: (x * c + z * s, -x * s + z * c)  # noqa: E731
    back = lambda a, b: (a * c - b * s, a * s + b * c)  # noqa: E731
    return fwd, back


def peak_height(heights, bins=0.02):
    # padded so a perfectly flat input (synthetic meshes) still gets bins
    hist, edges = np.histogram(heights, bins=np.arange(heights.min() - bins, heights.max() + 2 * bins, bins))
    i = int(np.argmax(hist))
    near = heights[(heights >= edges[i] - 0.03) & (heights <= edges[i + 1] + 0.03)]
    return float(np.median(near)), int(hist[i])


def heading(points_n, weights):
    phi = np.arctan2(points_n[:, 2], points_n[:, 0]) * 4  # x4 folds the four wall directions together
    return math.atan2((np.sin(phi) * weights).sum(), (np.cos(phi) * weights).sum()) / 4


def find_planes(coord, sign_ok, min_count):
    """Offsets (along one axis, one facing direction) where many vertical points line up."""
    c = coord[sign_ok]
    if len(c) < min_count:
        return []
    hist, edges = np.histogram(c, bins=np.arange(c.min() - 0.05, c.max() + 0.06, 0.01))
    smooth = np.convolve(hist, np.ones(5), mode='same')
    peaks = []
    for i in np.argsort(smooth)[::-1]:
        if smooth[i] < min_count * 0.3:
            break
        if all(abs(i - j) > 15 for j in peaks):  # 15 cm apart
            peaks.append(i)
    out = []
    for i in peaks:
        centre = (edges[i] + edges[i + 1]) / 2
        near = c[np.abs(c - centre) < WALL_BAND]
        if len(near) >= min_count:
            out.append(float(np.median(near)))
    return out


def plane_basis(normal):
    """Two unit vectors spanning the plane with this normal."""
    e1 = np.cross(normal, [0.0, 1.0, 0.0] if abs(normal[1]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(normal, e1)


def outline(points, normal, cell=0.05):
    """Observed outline of a planar patch, as a closed polygon of world points."""
    e1, e2 = plane_basis(normal)
    origin = points.mean(0)
    u, w = (points - origin) @ e1, (points - origin) @ e2
    u0, w0 = u.min(), w.min()
    grid = np.zeros((int((w.max() - w0) / cell) + 3, int((u.max() - u0) / cell) + 3), np.uint8)
    grid[((w - w0) / cell).astype(int) + 1, ((u - u0) / cell).astype(int) + 1] = 255
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.5, True).reshape(-1, 2)
    return [origin + ((x - 1) * cell + u0) * e1 + ((y - 1) * cell + w0) * e2 for x, y in c]


def ceiling_faces(v, n, h, floor_y, density, max_faces=16):
    """Large connected planar downward-facing patches above 1 m: flat ceilings, slopes, hip triangles."""
    cand = (n[:, 1] < -0.25) & (h > 1.0)  # pitch up to ~75 degrees
    pts = v[cand]
    min_pts = MIN_FACE_AREA * density
    faces = []
    while len(pts) > min_pts and len(faces) < max_faces:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        plane, idx = pc.segment_plane(0.03, 3, 1000)
        idx = np.asarray(idx)
        if len(idx) < min_pts:
            break
        labels = np.asarray(pc.select_by_index(idx).cluster_dbscan(eps=0.12, min_points=10))
        if labels.max() < 0 or np.bincount(labels[labels >= 0]).max() < min_pts:
            pts = np.delete(pts, idx, axis=0)  # scattered coplanar points: not a surface, don't refit
            continue
        keep = idx[labels == np.bincount(labels[labels >= 0]).argmax()]
        normal, d = np.array(plane[:3]), plane[3]
        if normal[1] > 0:
            normal, d = -normal, -d  # face down, into the room
        inl = pts[keep]
        pitch = float(np.degrees(np.arccos(min(1.0, -normal[1]))))
        faces.append({'normal': normal, 'd': float(d), 'points': inl, 'pitch': pitch,
                      'kind': 'flat' if pitch < FLAT_PITCH else 'slope', 'area': len(keep) / density,
                      'outline': outline(inl, normal), 'y': (float(inl[:, 1].min()), float(inl[:, 1].max()))})
        pts = np.delete(pts, keep, axis=0)
    return faces


def face_edges(faces, near=0.15, min_len=0.2):
    """Lines where two observed faces meet: peak (both fall away from it) or valley (both rise)."""
    edges = []
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            fi, fj = faces[i], faces[j]
            direction = np.cross(fi['normal'], fj['normal'])
            if np.linalg.norm(direction) < 0.2:  # nearly parallel
                continue
            direction /= np.linalg.norm(direction)
            # a point on both planes: least squares on the two plane equations plus "closest to origin"
            a_mat = np.array([fi['normal'], fj['normal'], direction])
            point = np.linalg.solve(a_mat, [-fi['d'], -fj['d'], 0.0])
            spans = []
            for f in (fi, fj):
                rel = f['points'] - point
                dist = np.linalg.norm(rel - np.outer(rel @ direction, direction), axis=1)
                close = rel[dist < near] @ direction
                if len(close) < 20:
                    break
                spans.append((np.percentile(close, 2), np.percentile(close, 98)))
            if len(spans) < 2:
                continue
            lo, hi = max(spans[0][0], spans[1][0]), min(spans[0][1], spans[1][1])
            if hi - lo < min_len:
                continue
            mid = point + direction * (lo + hi) / 2
            rises = []
            for f in (fi, fj):  # from the edge into the face, perpendicular to the edge, within the plane
                inward = np.cross(f['normal'], direction)
                if np.dot(f['points'].mean(0) - mid, inward) < 0:
                    inward = -inward
                rises.append(inward[1])
            kind = 'peak' if max(rises) < -0.05 else 'valley' if min(rises) > 0.05 else 'step'
            edges.append({'faces': [i, j], 'kind': kind,
                          'from': point + direction * lo, 'to': point + direction * hi,
                          'dihedralDeg': float(np.degrees(np.arccos(np.clip(np.dot(fi['normal'], fj['normal']), -1, 1))))})
    return edges


def face_height(face, x, z, margin=0.3):
    """Height of a face's plane above (x, z) when that spot lies under the face's observed outline."""
    nx, ny, nz = face['normal']
    if abs(ny) < 1e-3:
        return None
    poly = np.array([[p[0], p[2]] for p in face['outline']], np.float32)
    if cv2.pointPolygonTest(poly, (float(x), float(z)), True) < -margin:
        return None
    return -(face['d'] + nx * x + nz * z) / ny


def runs(values, gap):
    """Split sorted values into runs wherever consecutive values are more than gap apart."""
    v = np.sort(values)
    cuts = np.where(np.diff(v) > gap)[0]
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(v) - 1]
    return [(v[s], v[e]) for s, e in zip(starts, ends)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bundle')
    ap.add_argument('--mesh', required=True, help='scan_world.ply from reconstruct_bundle.py')
    ap.add_argument('--name', default='Room')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--plan', help='draw the detected layout (top view) to this PNG')
    ap.add_argument('--verbose', action='store_true', help='list every wall candidate and why it was kept or dropped')
    a = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(a.mesh)
    mesh.compute_vertex_normals()
    v, n = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    density = len(v) / max(mesh.get_surface_area(), 1e-6)  # vertices per m^2

    floor_y, _ = peak_height(v[n[:, 1] > 0.95, 1])
    h = v[:, 1] - floor_y
    faces = ceiling_faces(v, n, h, floor_y, density)
    flats = [f for f in faces if f['kind'] == 'flat' and f['area'] >= MIN_FLAT_AREA]
    ceiling_y = max(flats, key=lambda f: f['area'])['y'][1] if flats else None
    edges = face_edges(faces)

    vert = (np.abs(n[:, 1]) < 0.15) & (h > 0.05) & (h < 3.5)
    theta = heading(n[vert], np.ones(vert.sum()))
    fwd, back = to_frame(theta)
    pa, pb = fwd(v[:, 0], v[:, 2])
    na, nb = fwd(n[:, 0], n[:, 2])
    min_count = int(MIN_WALL_AREA * density)

    walls = []  # in the heading frame: axis 'a' walls are planes a = const facing +/-a
    for axis, coord, along, ncomp in (('a', pa, pb, na), ('b', pb, pa, nb)):
        for sign in (1, -1):
            facing = vert & (ncomp * sign > 0.9)
            planes = find_planes(coord, facing, min_count)
            a.verbose and print(f'{axis}{"+" if sign > 0 else "-"}: {facing.sum()} facing points, planes at {np.round(planes, 2).tolist()}')
            for offset in planes:
                inl = facing & (np.abs(coord - offset) < WALL_BAND)
                for u0, u1 in runs(along[inl], 0.5):
                    seg = inl & (along >= u0) & (along <= u1)
                    label = f'{axis}{"+" if sign > 0 else "-"} at {offset:+.2f}, u {u0:+.2f}..{u1:+.2f}'
                    if u1 - u0 < MIN_WALL_LENGTH or seg.sum() < min_count * 0.5:
                        a.verbose and print(f'  drop {label}: small ({u1 - u0:.2f} m, {seg.sum()} points)')
                        continue
                    top, bottom = np.percentile(h[seg], 95), np.percentile(h[seg], 5)
                    if top < MIN_WALL_TOP:
                        a.verbose and print(f'  drop {label}: low (top {top:.2f} m)')
                        continue
                    if bottom > MAX_WALL_BOTTOM:
                        a.verbose and print(f'  drop {label}: does not reach the floor (bottom {bottom:.2f} m)')
                        continue
                    # outermost? solid vertical surface beyond the plane within its span
                    beyond = vert & (along >= u0) & (along <= u1) & ((coord - offset) * -sign > BEHIND)
                    covered = np.unique(np.floor(along[beyond] / GRID)).size * GRID / (u1 - u0)
                    if covered > 0.5:
                        a.verbose and print(f'  drop {label}: surfaces behind it over {covered:.0%} of its length')
                        continue
                    a.verbose and print(f'  keep {label}: top {top:.2f} m, behind {covered:.0%}, {seg.sum()} points')
                    walls.append({'axis': axis, 'sign': sign, 'offset': offset, 'u0': float(u0), 'u1': float(u1),
                                  'points': int(seg.sum()), 'top': float(top), 'openings': []})

    # merge collinear pieces split by a gap, and record the gaps the scan sees through
    walls.sort(key=lambda w: (w['axis'], w['sign'], round(w['offset'], 2), w['u0']))
    merged = []
    for w in walls:
        prev = merged[-1] if merged else None
        if prev and prev['axis'] == w['axis'] and prev['sign'] == w['sign'] and abs(prev['offset'] - w['offset']) < 0.06 \
                and w['u0'] - prev['u1'] < 2.5:
            g0, g1 = prev['u1'], w['u0']
            coord = pa if w['axis'] == 'a' else pb
            along = pb if w['axis'] == 'a' else pa
            through = (along > g0 + 0.05) & (along < g1 - 0.05) & ((coord - w['offset']) * -w['sign'] > 0.3) & (h > 0.1)
            if through.sum() > 0.05 * density:
                width = g1 - g0
                prev['openings'].append({'u0': g0, 'u1': g1, 'kind': 'door' if DOOR_WIDTH[0] <= width <= DOOR_WIDTH[1] else 'opening'})
            prev['u1'] = w['u1']
            prev['points'] += w['points']
            prev['top'] = max(prev['top'], w['top'])
        else:
            merged.append(dict(w))
    walls = merged

    # snap ends into corners with perpendicular walls
    for w in walls:
        for other in walls:
            if other['axis'] == w['axis']:
                continue
            for end in ('u0', 'u1'):
                if abs(w[end] - other['offset']) < SNAP and other['u0'] - SNAP <= w['offset'] <= other['u1'] + SNAP:
                    w[end] = other['offset']
                    other['u0'] = min(other['u0'], w['offset'])
                    other['u1'] = max(other['u1'], w['offset'])

    tops = [f['y'][1] - floor_y for f in faces] + [w['top'] for w in walls]
    height = (ceiling_y - floor_y) if ceiling_y else max([2.4] + tops)

    def wall_profile(w, step=0.25):
        """World points along the wall's top: where the lowest ceiling face above it meets it."""
        profile = []
        for u in np.linspace(w['u0'], w['u1'], max(2, int((w['u1'] - w['u0']) / step) + 1)):
            x, z = back(*((w['offset'], u) if w['axis'] == 'a' else (u, w['offset'])))
            ys = [y for y in (face_height(f, x, z) for f in faces) if y is not None and y > floor_y + w['top'] - 0.1]
            if ys:
                profile.append([float(x), float(min(ys)), float(z)])
        return profile

    def world_point(axis, offset, u, y):
        a_, b_ = (offset, u) if axis == 'a' else (u, offset)
        x, z = back(a_, b_)
        return [x, y, z]

    def world_dir(axis, sign):
        """(wall direction u, inward normal) in world x/z, with u x up = normal."""
        na_, nb_ = (sign, 0.0) if axis == 'a' else (0.0, sign)
        nx, nz = back(na_, nb_)
        return np.array([nz, 0.0, -nx]), np.array([nx, 0.0, nz])

    def transform(center, u, normal):
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = u, [0, 1, 0], normal, center
        return [float(x) for x in m.T.reshape(-1)]

    out_walls, doors, openings = [], [], []
    for w in walls:
        u, normal = world_dir(w['axis'], w['sign'])
        mid = (w['u0'] + w['u1']) / 2
        length = w['u1'] - w['u0']
        profile = wall_profile(w)
        wh = min(p[1] for p in profile) - floor_y if profile else height
        w['height'], w['profile'] = wh, profile
        ident = str(uuid.uuid4()).upper()
        conf = 'high' if w['points'] > 4 * min_count and w['top'] > 1.9 else 'medium' if w['points'] > 2 * min_count else 'low'
        out_walls.append({'identifier': ident, 'transform': transform(world_point(w['axis'], w['offset'], mid, floor_y + wh / 2), u, normal),
                          'dimensions': [float(length), float(wh), 0.0], 'confidence': {conf: {}}, 'category': {'wall': {}},
                          'detectedBy': 'rig_layout', 'heightObserved': round(w['top'], 2),
                          'topProfile': profile})
        for o in w['openings']:
            om = (o['u0'] + o['u1']) / 2
            oh = min(DOOR_HEIGHT, wh) if o['kind'] == 'door' else wh
            rec = {'identifier': str(uuid.uuid4()).upper(), 'parentIdentifier': ident,
                   'transform': transform(world_point(w['axis'], w['offset'], om, floor_y + oh / 2), u, normal),
                   'dimensions': [float(o['u1'] - o['u0']), float(oh), 0.0], 'confidence': {'low': {}},
                   'category': {'door': {'isOpen': True}} if o['kind'] == 'door' else {'opening': {}}}
            (doors if o['kind'] == 'door' else openings).append(rec)

    # floor outline: footprint of everything up to 2 m plus the walls, closed and with holes filled
    # (furniture stands inside the room; the spot the operator stood on was never seen)
    floor_pts = (h > -0.05) & (h < 2.0)
    a0, b0 = pa.min(), pb.min()
    W, H = int((pa.max() - a0) / GRID) + 3, int((pb.max() - b0) / GRID) + 3
    grid = np.zeros((H, W), np.uint8)
    grid[((pb[floor_pts] - b0) / GRID).astype(int), ((pa[floor_pts] - a0) / GRID).astype(int)] = 255
    for w in walls:
        for u in np.arange(w['u0'], w['u1'], GRID / 2):
            a_, b_ = (w['offset'], u) if w['axis'] == 'a' else (u, w['offset'])
            cv2.circle(grid, (int((a_ - a0) / GRID), int((b_ - b0) / GRID)), 1, 255, -1)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # outer outline only: holes drop out
    polygon = []
    if contours:
        c = max(contours, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 0.08 / GRID, True).reshape(-1, 2)
        for x_, y_ in c:
            x, z = back(a0 + x_ * GRID, b0 + y_ * GRID)
            polygon.append([float(x), 0.0, float(z)])

    # objects: clusters of what is not floor, wall or ceiling
    wall_like = np.zeros(len(v), bool)
    for w in walls:
        coord, along = (pa, pb) if w['axis'] == 'a' else (pb, pa)
        wall_like |= (np.abs(coord - w['offset']) < 0.08) & (along >= w['u0'] - 0.1) & (along <= w['u1'] + 0.1)
    for f in faces:
        wall_like |= (np.abs(v @ f['normal'] + f['d']) < 0.06) & (h > 1.0)
    rest = (h > 0.05) & (h < height - 0.15) & ~wall_like
    objects = []
    if rest.sum() > 100:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.c_[pa[rest], h[rest], pb[rest]])).voxel_down_sample(0.04)
        labels = np.asarray(pc.cluster_dbscan(eps=0.09, min_points=25))
        pts = np.asarray(pc.points)
        for k in range(labels.max() + 1):
            q = pts[labels == k]
            lo, hi = np.percentile(q, 2, axis=0), np.percentile(q, 98, axis=0)
            size = hi - lo
            if size[1] < 0.2 or size[0] < 0.15 or size[2] < 0.15 or size[0] * size[2] < 0.05 or max(size[0], size[2]) > 3.5:
                continue
            bottom = 0.0 if lo[1] < 0.3 else lo[1]
            cx, cz = back((lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2)
            right, front = world_dir('b', 1)  # box axes follow the heading frame
            objects.append({'identifier': str(uuid.uuid4()).upper(), 'category': {'object': {}}, 'confidence': {'low': {}},
                            'transform': transform([cx, floor_y + (bottom + hi[1]) / 2, cz], right, front),
                            'dimensions': [float(size[0]), float(hi[1] - bottom), float(size[2])], 'points': int(len(q))})
        objects.sort(key=lambda o: -o['points'])
        objects = objects[:40]

    room = {'identifier': str(uuid.uuid4()).upper(), 'version': 2, 'detectedBy': 'kinect-rig rig_layout.py',
            'walls': out_walls, 'doors': doors, 'windows': [], 'openings': openings, 'objects': objects,
            'floors': [{'identifier': str(uuid.uuid4()).upper(), 'transform': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, floor_y, 0, 1],
                        'dimensions': [0, 0, 0], 'polygonCorners': polygon, 'confidence': {'medium': {}}, 'category': {'floor': {}}}],
            'ceilingHeight': None if ceiling_y is None else float(ceiling_y - floor_y),
            # not part of RoomPlan's format: the ceiling as observed planar faces and the edges between them
            'ceilingFaces': [{'kind': f['kind'], 'normal': [float(x) for x in f['normal']], 'd': f['d'],
                              'pitchDeg': round(f['pitch'], 1), 'areaObserved': round(f['area'], 2),
                              'heightRange': [round(f['y'][0] - floor_y, 3), round(f['y'][1] - floor_y, 3)],
                              'outline': [[float(c) for c in p] for p in f['outline']]} for f in faces],
            'ceilingEdges': [{'faces': e['faces'], 'kind': e['kind'], 'dihedralDeg': round(e['dihedralDeg'], 1),
                              'from': [float(c) for c in e['from']], 'to': [float(c) for c in e['to']]} for e in edges],
            'notes': ['walls assumed to meet at right angles', 'windows not detected',
                      'ceilingFaces: planes normal . p + d = 0 facing into the room (meters, bundle world); outline = observed extent',
                      'walls[].topProfile: where the lowest ceiling face meets the wall; dimensions[1] is its lowest point',
                      f'flat ceiling at {ceiling_y - floor_y:.2f} m' if ceiling_y else 'no flat ceiling of 1 m^2 or more seen']}
    os.makedirs(os.path.join(a.bundle, 'rooms'), exist_ok=True)
    rel = f'rooms/room-{a.index}.json'
    with open(os.path.join(a.bundle, rel), 'w') as f:
        json.dump(room, f, indent=1)

    manifest_path = os.path.join(a.bundle, 'manifest.json')
    manifest = json.load(open(manifest_path))
    record = {'index': a.index, 'name': a.name, 'file': rel, 'usdz': '',
              'capturedAt': manifest.get('createdAt') or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
              'walls': len(out_walls), 'doors': len(doors), 'windows': 0, 'openings': len(openings), 'objects': len(objects)}
    manifest['rooms'] = [r for r in manifest.get('rooms', []) if r['index'] != a.index] + [record]
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'floor {floor_y:+.3f} m, ' + (f'flat ceiling {ceiling_y - floor_y:.2f} m above it' if ceiling_y else 'no flat ceiling')
          + f', heading {math.degrees(theta):.1f} deg; {len(faces)} ceiling faces, {len(edges)} edges between them')
    for i, f in enumerate(faces):
        az = (math.degrees(math.atan2(f['normal'][2], f['normal'][0]) - theta)) % 360
        print(f"  face {i}: {f['kind']}, pitch {f['pitch']:.0f} deg, facing {az:.0f} deg from the heading, "
              f"{f['area']:.1f} m^2 observed, {f['y'][0] - floor_y:.2f}-{f['y'][1] - floor_y:.2f} m above the floor")
    for e in edges:
        print(f"  edge faces {e['faces']}: {e['kind']}, {np.linalg.norm(e['to'] - e['from']):.2f} m, "
              f"heights {e['from'][1] - floor_y:.2f} → {e['to'][1] - floor_y:.2f} m, dihedral {e['dihedralDeg']:.0f} deg")
    print(f'{len(out_walls)} walls, {len(doors)} doors, {len(openings)} openings, {len(objects)} objects, '
          f'floor outline {len(polygon)} corners → {os.path.join(a.bundle, rel)}')
    for w in walls:
        print(f"  wall {w['axis']}{'+' if w['sign'] > 0 else '-'} at {w['offset']:+.2f}: {w['u1'] - w['u0']:.2f} m long, "
              + (f"top {w['height']:.2f}-{max(p[1] for p in w['profile']) - floor_y:.2f} m" if w['profile']
                 else f"top {w['height']:.2f} m (no ceiling face seen above)")
              + f", observed to {w['top']:.2f} m, {len(w['openings'])} openings")
    if a.plan:
        draw(a.plan, pa, pb, h, walls, polygon and np.array([fwd(p[0], p[2]) for p in polygon]), objects, fwd, faces, edges)


def draw(path, pa, pb, h, walls, poly, objects, fwd, faces=(), edges=(), ppm=100):
    lo, hi = np.array([pa.min(), pb.min()]) - 0.3, np.array([pa.max(), pb.max()]) + 0.3
    W, H = (hi - lo) * ppm
    img = np.full((int(H), int(W), 3), 24, np.uint8)
    P = lambda a_, b_: (int((a_ - lo[0]) * ppm), int((b_ - lo[1]) * ppm))  # noqa: E731
    xs = np.clip(((pa - lo[0]) * ppm).astype(int), 0, img.shape[1] - 1)
    ys = np.clip(((pb - lo[1]) * ppm).astype(int), 0, img.shape[0] - 1)
    order = np.argsort(h)  # higher surfaces drawn last, brighter
    img[ys[order], xs[order]] = np.clip(60 + h[order] * 60, 60, 200).astype(np.uint8)[:, None]
    if poly is not None and len(poly):
        cv2.polylines(img, [np.array([P(*p) for p in poly], np.int32)], True, (90, 160, 90), 2)
    overlay = img.copy()
    for f in faces:  # flat faces blue, slopes purple
        pts = np.array([P(*fwd(p[0], p[2])) for p in f['outline']], np.int32)
        cv2.fillPoly(overlay, [pts], (170, 110, 60) if f['kind'] == 'flat' else (150, 60, 160))
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    for i, f in enumerate(faces):
        c = np.mean([fwd(p[0], p[2]) for p in f['outline']], axis=0)
        cv2.putText(img, f"{i}: {f['pitch']:.0f} deg", P(*c), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 190, 245), 1)
    for e in edges:
        color = (80, 220, 255) if e['kind'] == 'peak' else (255, 220, 80) if e['kind'] == 'valley' else (180, 180, 180)
        cv2.line(img, P(*fwd(e['from'][0], e['from'][2])), P(*fwd(e['to'][0], e['to'][2])), color, 3)
    for o in objects:
        m = np.array(o['transform']).reshape(4, 4).T
        ca, cb = fwd(m[0, 3], m[2, 3])
        sx, sz = o['dimensions'][0] / 2, o['dimensions'][2] / 2
        cv2.rectangle(img, P(ca - sx, cb - sz), P(ca + sx, cb + sz), (60, 170, 230), 2)
    for w in walls:
        p0 = P(*((w['offset'], w['u0']) if w['axis'] == 'a' else (w['u0'], w['offset'])))
        p1 = P(*((w['offset'], w['u1']) if w['axis'] == 'a' else (w['u1'], w['offset'])))
        cv2.line(img, p0, p1, (240, 240, 240), 4)
        for o in w['openings']:
            q0 = P(*((w['offset'], o['u0']) if w['axis'] == 'a' else (o['u0'], w['offset'])))
            q1 = P(*((w['offset'], o['u1']) if w['axis'] == 'a' else (o['u1'], w['offset'])))
            cv2.line(img, q0, q1, (255, 200, 80) if o['kind'] == 'door' else (200, 120, 255), 6)
    cv2.imwrite(path, img)


if __name__ == '__main__':
    main()
