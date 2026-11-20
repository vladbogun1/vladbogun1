"""Shared cover geometry: base plate, lid shell, hinge layouts."""
import math

import numpy as np

from geo import (CS, M, ROUND, apply, box, circle, mat_rot_x, mat_rot_y180, mat_translate, prism_x,
                 rect, rounded_rect_cs, teardrop, union, xcyl, zcyl)

# ------------------------------------------------------------ parameters
T = 1.2            # base plate thickness (sits under the button bezel)
HW = 12.0          # half width of the cover (24 mm)
YB = -11.2         # outer bottom edge of plate / lid
HOLE_R = 8.1       # 16.2 mm hole for a 16 mm button
FLANGE_R = 9.5     # button bezel radius (19 mm)
HEAD_Z = T + 4.0   # top of the button head above the panel
WALL = 1.2         # lid wall thickness (3 perimeters @0.4)
ZF = 7.2           # outer front face of the lid
FRONT = 1.2
RIM = T + 0.1      # lid rim height (0.1 mm air over the plate)
XIN = HW - WALL    # inner side wall
YIN_B = YB + WALL  # inner bottom wall
Y_CLEAR = FLANGE_R + 0.3   # anything at the top must stay above this
CORNER = 1.0


def base_plate(y_top):
    cs = rounded_rect_cs(-HW, HW, YB, y_top, CORNER)
    plate = M.extrude(cs, T)
    return plate - zcyl(HOLE_R, -1, T + 1)


def lid_shell(y_top, rim=RIM, rim_top=None):
    outer = M.extrude(rounded_rect_cs(-HW, HW, YB, y_top, CORNER), ZF - rim).translate((0, 0, rim))
    inner = box(-XIN, XIN, YIN_B, y_top - WALL, rim - 1, ZF - FRONT)
    lid = outer - inner
    # finger lip at the bottom front edge
    lip = M.extrude(rounded_rect_cs(-5, 5, YB - 1.6, YB + 1.0, 0.8), 1.2).translate((0, 0, ZF - 1.2))
    lid = lid + lip
    if rim_top is not None and rim_top > rim:
        lid = lid - box(-XIN, XIN, y_top - WALL - 0.01, y_top + 1, rim - 1, rim_top)
    return lid


def button_dummy():
    """Display-only 16 mm anti-vandal button + panel."""
    panel = box(-21, 21, -24, 24, -2.0, 0.0)
    flange = zcyl(FLANGE_R, T, T + 1.6, r2=FLANGE_R - 0.4)
    body = zcyl(FLANGE_R - 0.4, T + 1.6, HEAD_Z - 0.4, r2=7.6)
    cap = zcyl(6.6, HEAD_Z - 0.4, HEAD_Z)
    ring = zcyl(7.5, HEAD_Z - 0.45, HEAD_Z - 0.2) - zcyl(6.9, HEAD_Z - 1, HEAD_Z)
    btn = union(flange, body, cap) - ring
    return panel, btn, ring


# ------------------------------------------------------------ sweep cut
def sweep_cs(cs, y, z, a0, a1, step=3.0):
    """Union of a 2D section rotated about (y,z) by angles a0..a1 (deg)."""
    out = []
    n = max(1, int(math.ceil(abs(a1 - a0) / step)))
    for i in range(n + 1):
        a = a0 + (a1 - a0) * i / n
        out.append(cs.translate((-y, -z)).rotate(a).translate((y, z)))
    return CS.batch_boolean(out, __import__("manifold3d").OpType.Add)


# ------------------------------------------------------------ print poses
def P_lid_facedown(dx=30.0):
    """Lid printed front-face-down next to the base."""
    return mat_translate(dx, 0, ZF) @ mat_rot_y180()


def P_open(deg, y, z):
    """Pose where the lid is rotated open by `deg` about the hinge axis."""
    return mat_rot_x(-deg, y, z)


class Knuckle:
    """Barrel hinge: base knuckle(s) in the middle, lid ears at both ends.

    pip=True places the axis at ZF/2 so that the lid opened 180 deg lies
    face-down on the bed next to the base (print-in-place).
    """

    def __init__(self, R, xk=7.7, gax=0.3, g=0.35, pip=False, sweep=100, base_ranges=None,
                 ear_x0=None, z_h=None):
        self.R = R
        self.pip = pip
        self.z_h = z_h if z_h is not None else (ZF / 2 if pip else T + R - 0.5)
        self.y_h = Y_CLEAR + R
        self.y_top = self.y_h + R
        self.y_pm = self.y_h - R - 0.3
        self.xk = xk
        self.gax = gax
        self.g = g
        self.sweep = sweep
        self.base_ranges = base_ranges or [(-xk, xk)]
        self.ear_x0 = ear_x0 if ear_x0 is not None else xk + gax

    # base -------------------------------------------------------------
    def support_cs(self):
        y, z, R = self.y_h, self.z_h, self.R
        tab = rect(self.y_pm - 1.0, y, 0, T)
        skirt = CS.batch_hull([circle(R, y, z), rect(y - 0.85 * R, y + 0.45 * R, 0, T)])
        return tab + skirt

    def base(self):
        b = base_plate(self.y_pm)
        cs = self.support_cs()
        tab_all = prism_x(rect(self.y_pm - 1.0, self.y_h, 0, T), -self.xk, self.xk)
        parts = [b, tab_all]
        for x0, x1 in self.base_ranges:
            parts.append(prism_x(cs, x0, x1))
        return union(*parts)

    # lid --------------------------------------------------------------
    def ear_cs(self):
        y, z, R = self.y_h, self.z_h, self.R
        return circle(R, y, z) + rect(y - R, self.y_top, z, ZF)

    def lid(self, ears=True):
        lid = lid_shell(self.y_top)
        if ears:
            cs = self.ear_cs()
            lid = union(lid, prism_x(cs, self.ear_x0, HW), prism_x(cs, -HW, -self.ear_x0))
        x_c = self.ear_x0 - 1e-3
        cut = prism_x(circle(self.R + self.g, self.y_h, self.z_h), -x_c, x_c)
        sw = sweep_cs(self.support_cs().offset(self.g, ROUND), self.y_h, self.z_h, 0, self.sweep)
        cut = cut + prism_x(sw, -x_c, x_c)
        # behind/below the axis keep only the barrel -> lid never sweeps into the panel
        corner = rect(self.y_h, self.y_top + 1, -1, self.z_h) - circle(self.R, self.y_h, self.z_h)
        cut = cut + prism_x(corner, -HW - 1, HW + 1)
        return lid - cut

    def x_hole(self, r, x0, x1, tip):
        return prism_x(teardrop(r, self.y_h, self.z_h, tip), x0, x1)

    def motion(self, max_open):
        return {"type": "rotate", "y": self.y_h, "z": self.z_h, "max": max_open}


class Posts:
    """Base carries two outer posts (with a bar or magnets); lid carries the
    central element (hooks / knuckle)."""

    def __init__(self, Rp, xp, gax=0.4, g=0.35, sweep=100, y_h=None):
        self.R = Rp
        self.xp = xp
        self.gax = gax
        self.g = g
        self.sweep = sweep
        self.z_h = T + Rp - 0.5
        self.y_h = y_h if y_h is not None else Y_CLEAR + Rp
        self.y_pm = self.y_h - Rp - 0.3

    def post_cs(self):
        y, z, R = self.y_h, self.z_h, self.R
        return rect(self.y_pm - 1.0, y, 0, T) + CS.batch_hull([circle(R, y, z), rect(y - 0.85 * R, y + 0.45 * R, 0, T)])

    def base(self):
        cs = self.post_cs()
        return union(base_plate(self.y_pm), prism_x(cs, self.xp, HW), prism_x(cs, -HW, -self.xp))

    def lid_cut(self):
        sw = sweep_cs(self.post_cs().offset(self.g, ROUND), self.y_h, self.z_h, 0, self.sweep)
        return prism_x(sw, self.xp - self.gax, HW + 1) + prism_x(sw, -HW - 1, -(self.xp - self.gax))

    def motion(self, max_open):
        return {"type": "rotate", "y": self.y_h, "z": self.z_h, "max": max_open}


class Flex:
    """Flexure hinge. Geometry lives in the FLAT frame (= print pose, lid
    opened 90 deg, its top wall lying on the bed). Closing bends the strip
    by alpha = 90 deg."""

    def __init__(self, L, t, zm, y_p=9.6, xr=((-10.6, 10.6),)):
        self.L, self.t, self.zm, self.y_p = L, t, zm, y_p
        self.xr = [list(r) for r in xr]
        self.F = (y_p + L, zm)

    def end_pose(self, alpha_deg):
        a = math.radians(alpha_deg)
        y0, z0 = self.y_p, self.zm
        if abs(a) < 1e-9:
            return (y0 + self.L, z0), 0.0
        rho = self.L / a
        return (y0 + rho * math.sin(a), z0 + rho * (1 - math.cos(a))), alpha_deg

    def lid_matrix(self, alpha_deg):
        (ey, ez), ang = self.end_pose(alpha_deg)
        return mat_translate(0, ey, ez) @ mat_rot_x(ang) @ mat_translate(0, -self.F[0], -self.F[1])

    @property
    def closed(self):
        (ey, ez), _ = self.end_pose(90)
        return ey, ez

    def y_top(self):
        ey, ez = self.closed
        return ey + self.zm

    def lid_flat(self, extra_cut=None):
        ey, ez = self.closed
        lid_c = lid_shell(self.y_top(), RIM, ez)
        inv = np.linalg.inv(self.lid_matrix(90))
        lid_f = apply(lid_c, inv)
        return lid_f

    def strip(self, overlap=0.4):
        parts = [box(x0, x1, self.y_p - overlap, self.F[0] + overlap, self.zm - self.t / 2, self.zm + self.t / 2)
                 for x0, x1 in self.xr]
        return union(*parts)

    def motion(self):
        return {"type": "flex", "yP": self.y_p, "zm": self.zm, "L": self.L, "t": self.t,
                "xr": self.xr, "Fy": self.F[0], "Fz": self.F[1]}
