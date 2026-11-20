"""Geometry helpers on top of manifold3d.

Coordinate frame (assembled, lid closed):
  X - across the cover (width), Y - up along the panel, Z - out of the panel.
  Panel surface is z = 0, button axis is the Z axis through (0, 0).
All dimensions in millimetres.
"""
import math
import struct

import numpy as np
import manifold3d as m3

M = m3.Manifold
CS = m3.CrossSection
ROUND = m3.JoinType.Round
MITER = m3.JoinType.Miter

SEG = 64  # circular segments for small round features
m3.set_circular_segments(SEG)


# ---------------------------------------------------------------- primitives
def box(x0, x1, y0, y1, z0, z1):
    return M.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def zcyl(r, z0, z1, x=0.0, y=0.0, r2=None, seg=SEG):
    return M.cylinder(z1 - z0, r, r if r2 is None else r2, seg).translate((x, y, z0))


def prism_x(cs, x0, x1):
    """Extrude a 2D cross-section given in (y, z) along X from x0 to x1."""
    if x1 < x0:
        x0, x1 = x1, x0
    mat = np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    return M.extrude(cs, x1 - x0).transform(mat)


def circle(r, y, z, seg=SEG):
    return CS.circle(r, seg).translate((y, z))


def rect(y0, y1, z0, z1):
    return CS.square((y1 - y0, z1 - z0)).translate((y0, z0))


def teardrop(r, y, z, tip=(0, 1), seg=SEG):
    """Circle with a 45 deg pointed tip (support-free horizontal hole/pin)."""
    tx, tz = tip
    n = math.hypot(tx, tz)
    tx, tz = tx / n, tz / n
    d = r * math.sqrt(2)
    tri = CS.hull_points([(y + tx * d, z + tz * d), (y - tz * r * 0.99, z + tx * r * 0.99),
                          (y + tz * r * 0.99, z - tx * r * 0.99)])
    return CS.batch_hull([circle(r, y, z, seg), tri])


def xcyl(r, x0, x1, y, z, seg=SEG):
    return prism_x(circle(r, y, z, seg), x0, x1)


def xcone(r0, r1, x0, x1, y, z, seg=SEG):
    """Cone along X: radius r0 at x0, r1 at x1 (x0 may be > x1)."""
    lo, hi = (x0, x1) if x0 < x1 else (x1, x0)
    rlo, rhi = (r0, r1) if x0 < x1 else (r1, r0)
    c = M.cylinder(hi - lo, rlo, rhi, seg)
    mat = np.array([[0, 0, 1, lo], [1, 0, 0, y], [0, 1, 0, z]], dtype=float)
    return c.transform(mat)


def union(*parts):
    parts = [p for p in parts if p is not None]
    return M.batch_boolean(parts, m3.OpType.Add) if len(parts) > 1 else parts[0]


def rot_about_x(solid, deg, y, z):
    return solid.translate((0, -y, -z)).rotate((deg, 0, 0)).translate((0, y, z))


def rounded_rect_cs(x0, x1, y0, y1, r):
    return CS.square((x1 - x0 - 2 * r, y1 - y0 - 2 * r)).translate((x0 + r, y0 + r)).offset(r, ROUND)


# ------------------------------------------------------------ 4x4 matrices
def mat_rot_x(deg, y=0.0, z=0.0):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    T1 = np.eye(4); T1[1, 3] = -y; T1[2, 3] = -z
    R = np.eye(4); R[1, 1] = c; R[1, 2] = -s; R[2, 1] = s; R[2, 2] = c
    T2 = np.eye(4); T2[1, 3] = y; T2[2, 3] = z
    return T2 @ R @ T1


def mat_rot_y180():
    R = np.eye(4); R[0, 0] = -1; R[2, 2] = -1
    return R


def mat_translate(x=0.0, y=0.0, z=0.0):
    T = np.eye(4); T[0, 3] = x; T[1, 3] = y; T[2, 3] = z
    return T


def apply(solid, mat4):
    return solid.transform(np.asarray(mat4)[:3, :])


# ---------------------------------------------------------------- export
def write_stl(solid, path):
    mesh = solid.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
    t = np.asarray(mesh.tri_verts)
    tri = v[t]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    n = (n / ln).astype(np.float32)
    with open(path, "wb") as f:
        f.write(b"button-cover-hinges".ljust(80, b" "))
        f.write(struct.pack("<I", len(t)))
        rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"] = n
        rec["v"] = tri
        f.write(rec.tobytes())
    return len(t)
