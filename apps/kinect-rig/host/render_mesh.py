#!/usr/bin/env python3
"""Render a reconstructed room mesh to preview images (no GPU or display needed).

usage: render_mesh.py <mesh.ply> -o PREFIX [--cut 1.6] [--band LOW HIGH]

Expects the bundle's world: meters, +Y up. Heights are measured from the main floor, found as the
most common height of upward-facing surfaces. Writes
  PREFIX-angle.jpg   angled view, photo colors
  PREFIX-top.jpg     straight down, photo colors
  PREFIX-height.jpg  straight down, colored by height above the floor; --band (meters relative to
                     the floor) is painted magenta to locate a suspect layer
Everything above --cut meters over the floor is removed so the views look into the room.
"""
import argparse

import cv2
import numpy as np
import open3d as o3d


def floor_height(mesh):
    mesh.compute_vertex_normals()
    v, n = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    up = v[n[:, 1] > 0.95, 1]
    hist, edges = np.histogram(up, bins=np.arange(up.min(), up.max() + 0.02, 0.02))
    i = int(np.argmax(hist))
    return float(np.median(up[(up >= edges[i] - 0.03) & (up <= edges[i + 1] + 0.03)]))


def render(mesh, size, view, extent, center):
    """Fill every triangle far-to-near with its mean vertex color (painter's algorithm).
    Open3D's offscreen renderer returns blank frames once a scene passes ~65k vertices
    (0.20, Vulkan, NVIDIA 560), so previews are drawn with OpenCV instead."""
    w, h = size
    v = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    colors = (np.asarray(mesh.vertex_colors)[tris].mean(1)[:, ::-1] * 255).astype(np.uint8)  # BGR
    if view == 'top':
        scale = 0.92 * min(w / extent[0], h / extent[2])
        uv = np.c_[(v[:, 0] - center[0]) * scale + w / 2, (v[:, 2] - center[2]) * scale + h / 2]
        depth = -v[:, 1]  # looking down: higher surfaces are nearer
        visible = np.ones(len(tris), bool)
    else:
        eye = center + np.array([-0.55, 0.75, 0.9]) * np.linalg.norm(extent) * 0.5
        fwd = (center - eye) / np.linalg.norm(center - eye)
        right = np.cross(fwd, [0, 1, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        rel = v - eye
        x, y, z = rel @ right, rel @ up, rel @ fwd
        f = 0.5 * h / np.tan(np.radians(55) / 2)
        uv = np.c_[f * x / np.maximum(z, 1e-3) + w / 2, -f * y / np.maximum(z, 1e-3) + h / 2]
        depth = z
        visible = (z[tris] > 0.1).all(1)
    img = np.full((h, w, 3), (28, 23, 23), np.uint8)
    order = np.argsort(-depth[tris].mean(1))
    pts = np.round(uv[tris] * 16).astype(np.int32)  # 4 bits of subpixel precision
    for i in order:
        if visible[i]:
            cv2.fillConvexPoly(img, pts[i], colors[i].tolist(), lineType=cv2.LINE_AA, shift=4)
    return img


def label(img, lines):
    for i, text in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mesh')
    ap.add_argument('-o', required=True, help='output prefix')
    ap.add_argument('--cut', type=float, default=1.6, help='remove everything this far above the floor (m)')
    ap.add_argument('--band', type=float, nargs=2, metavar=('LOW', 'HIGH'), help='height band to highlight (m from floor)')
    ap.add_argument('--size', type=int, nargs=2, default=(1600, 1200))
    a = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(a.mesh)
    floor = floor_height(mesh)
    v = np.asarray(mesh.vertices)
    mesh.remove_vertices_by_mask(v[:, 1] > floor + a.cut)
    v = np.asarray(mesh.vertices)
    lo, hi = v.min(0), v.max(0)
    center, extent = (lo + hi) / 2, hi - lo
    center[1] = floor

    size = tuple(a.size)
    info = f'floor at {floor:+.2f} m (world), cut {a.cut:.1f} m above it; extent {extent[0]:.1f} x {extent[2]:.1f} m'
    cv2.imwrite(f'{a.o}-angle.jpg', label(render(mesh, size, 'angle', extent, center), ['photo colors', info]))
    cv2.imwrite(f'{a.o}-top.jpg', label(render(mesh, size, 'top', extent, center), ['top view, photo colors', info]))

    rel = v[:, 1] - floor
    norm = np.clip((rel + 0.2) / (a.cut + 0.2), 0, 1)
    colors = cv2.applyColorMap((norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1] / 255.0
    lines = ['top view, color = height above floor', 'blue = floor level ... red = ' + f'{a.cut:.1f} m']
    if a.band:
        sel = (rel >= a.band[0]) & (rel <= a.band[1])
        colors[sel] = [1.0, 0.0, 1.0]
        lines.append(f'magenta = {a.band[0] * 100:+.0f} to {a.band[1] * 100:+.0f} cm ({sel.sum()} vertices)')
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    cv2.imwrite(f'{a.o}-height.jpg', label(render(mesh, size, 'top', extent, center), lines))
    print(f'floor {floor:+.3f} m → {a.o}-angle.jpg, -top.jpg, -height.jpg')


if __name__ == '__main__':
    main()
