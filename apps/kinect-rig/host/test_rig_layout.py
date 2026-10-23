#!/usr/bin/env python3
"""rig_layout.py on a synthetic hip-roof attic with a known answer.

5 x 4 m floor, 1.0 m knee walls all round, four 40-degree roof faces (two trapezoids, two hip
triangles) meeting in a 1 m ridge and four hips. Run: python3 test_rig_layout.py (needs open3d).
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
LENGTH, WIDTH, EAVE, PITCH = 5.0, 4.0, 1.0, 40.0


def attic_mesh():
    rise = WIDTH / 2 * math.tan(math.radians(PITCH))
    x0, x1, z0, z1 = -LENGTH / 2, LENGTH / 2, -WIDTH / 2, WIDTH / 2
    ridge_y, inset = EAVE + rise, WIDTH / 2  # equal pitches: hips run at 45 degrees in plan
    r0, r1 = (x0 + inset, ridge_y, 0.0), (x1 - inset, ridge_y, 0.0)
    e = {k: (x, EAVE, z) for k, (x, z) in {'a': (x0, z0), 'b': (x1, z0), 'c': (x1, z1), 'd': (x0, z1)}.items()}
    f = {k: (x, 0.0, z) for k, (x, z) in {'a': (x0, z0), 'b': (x1, z0), 'c': (x1, z1), 'd': (x0, z1)}.items()}
    quads = [(f['a'], f['d'], f['c'], f['b']),                      # floor, facing up
             (f['a'], f['b'], e['b'], e['a']), (f['b'], f['c'], e['c'], e['b']),
             (f['c'], f['d'], e['d'], e['c']), (f['d'], f['a'], e['a'], e['d']),  # knee walls, facing in
             (e['a'], e['b'], r1, r0), (e['c'], e['d'], r0, r1)]    # long roof faces
    tris = [(e['b'], e['c'], r1), (e['d'], e['a'], r0)]             # hip triangles
    mesh = o3d.geometry.TriangleMesh()
    for q in quads:
        part = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(q), o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))
        mesh += part
    for t in tris:
        mesh += o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(t), o3d.utility.Vector3iVector([[0, 1, 2]]))
    # make every face's normal point into the room (ray from the room's middle)
    tri = np.asarray(mesh.triangles)
    v = np.asarray(mesh.vertices)
    centre = np.array([0.0, EAVE * 0.6, 0.0])
    normals = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    flip = np.einsum('ij,ij->i', normals, centre - v[tri].mean(1)) < 0
    tri[flip] = tri[flip][:, ::-1]
    mesh.triangles = o3d.utility.Vector3iVector(tri)
    mesh = mesh.subdivide_midpoint(7)
    mesh.merge_close_vertices(1e-6)
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    return mesh, ridge_y


class HipRoofAttic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        bundle = os.path.join(cls.tmp.name, 'bundle')
        os.makedirs(bundle)
        json.dump({'format': 'house-capture/5', 'rooms': [], 'createdAt': '2026-01-01T00:00:00+00:00'},
                  open(os.path.join(bundle, 'manifest.json'), 'w'))
        mesh, cls.ridge_y = attic_mesh()
        path = os.path.join(cls.tmp.name, 'attic.ply')
        o3d.io.write_triangle_mesh(path, mesh)
        out = subprocess.run([sys.executable, os.path.join(HERE, 'rig_layout.py'), bundle, '--mesh', path],
                             capture_output=True, text=True)
        cls.log = out.stdout + out.stderr
        assert out.returncode == 0, cls.log
        cls.room = json.load(open(os.path.join(bundle, 'rooms', 'room-0.json')))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_four_roof_faces_at_the_pitch(self):
        faces = self.room['ceilingFaces']
        self.assertEqual(len(faces), 4, self.log)
        for f in faces:
            self.assertEqual(f['kind'], 'slope')
            self.assertAlmostEqual(f['pitchDeg'], PITCH, delta=1.0)
        self.assertIsNone(self.room['ceilingHeight'])  # no flat ceiling here

    def test_ridge_and_hips_are_peaks(self):
        edges = self.room['ceilingEdges']
        peaks = [e for e in edges if e['kind'] == 'peak']
        self.assertGreaterEqual(len(peaks), 5, self.log)  # 1 ridge + 4 hips (+ none of them valleys)
        self.assertFalse([e for e in edges if e['kind'] == 'valley'], self.log)
        ridge = [e for e in peaks if abs(e['from'][1] - e['to'][1]) < 0.05]
        self.assertTrue(ridge, self.log)
        self.assertAlmostEqual(ridge[0]['from'][1], self.ridge_y, delta=0.05)

    def test_knee_walls_follow_the_eave(self):
        walls = self.room['walls']
        self.assertEqual(len(walls), 4, self.log)
        for w in walls:
            self.assertAlmostEqual(w['dimensions'][1], EAVE, delta=0.06)
            self.assertTrue(w['topProfile'])
        lengths = sorted(round(w['dimensions'][0], 1) for w in walls)
        self.assertAlmostEqual(lengths[0], WIDTH, delta=0.1)
        self.assertAlmostEqual(lengths[-1], LENGTH, delta=0.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
