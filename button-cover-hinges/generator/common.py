"""Shared geometry of the FLAT cover: a frame around a flush/recessed 16 mm
button and a flat lid lying on top of it. Hinge layouts live here too.

Frame (assembled, lid closed): X across, Y up along the panel, Z out of the
panel. Panel surface z = 0, button axis = Z axis. Millimetres.
"""
import math

import numpy as np
import manifold3d as m3

from geo import (CS, M, ROUND, apply, box, circle, mat_rot_x, mat_rot_y180, mat_translate, prism_x,
                 rect, rounded_rect_cs, teardrop, union, zcyl)

# ------------------------------------------------------------ parameters
BTN_TOP = 1.0      # top of the button above the panel (flush / almost flush button)
T = 1.8            # frame thickness: must be > BTN_TOP (+0.5 mm air under the lid)
HOLE_R = 10.0      # 20 mm window around the button bezel (bezel up to ~19.4 mm)
HW = 13.0          # half width (26 mm)
YB = -11.8         # bottom edge of the frame
LIP = 1.2          # finger tab of the lid below the frame
GAP = 0.1          # air between frame and lid
LID = 1.4          # lid thickness
ZL0 = T + GAP      # lid underside
ZL1 = ZL0 + LID    # lid top = total height (3.3 mm)
ZH = ZL1 / 2       # hinge axis: barrel flush with the panel AND with the lid top
RB = ZH            # barrel radius (3.3 mm barrel)
Y_CLEAR = HOLE_R + 0.8   # hinge knuckles stay above the window
CORNER = 1.5
FLANGE_R = 9.5     # display-only button bezel


def base_plate(y_top):
    cs = rounded_rect_cs(-HW, HW, YB, y_top, CORNER)
    return M.extrude(cs, T) - zcyl(HOLE_R, -1, T + 1)


def lid_plate(y_top):
    body = M.extrude(rounded_rect_cs(-HW, HW, YB, y_top, CORNER), LID).translate((0, 0, ZL0))
    tab = M.extrude(rounded_rect_cs(-5, 5, YB - LIP, YB + 2, 0.8), LID).translate((0, 0, ZL0))
    return body + tab


def button_dummy():
    panel = box(-24, 24, -26, 26, -2.0, 0.0)
    bezel = zcyl(FLANGE_R, 0.0, BTN_TOP - 0.2, r2=FLANGE_R - 0.3)
    cap = zcyl(7.4, BTN_TOP - 0.2, BTN_TOP, r2=7.2)
    ring = zcyl(7.9, BTN_TOP - 0.25, BTN_TOP - 0.05) - zcyl(7.45, -1, 3)
    btn = union(bezel, cap) - ring
    return panel, btn, ring


def sweep_cs(cs, y, z, a0, a1, step=3.0):
    out = []
    n = max(1, int(math.ceil(abs(a1 - a0) / step)))
    for i in range(n + 1):
        a = a0 + (a1 - a0) * i / n
        out.append(cs.translate((-y, -z)).rotate(a).translate((y, z)))
    return CS.batch_boolean(out, m3.OpType.Add)


def P_lid_facedown(dx=32.0):
    """Lid printed top-face-down next to the base."""
    return mat_translate(dx, 0, ZL1) @ mat_rot_y180()


def P_open(deg, y, z):
    return mat_rot_x(-deg, y, z)


class Knuckle:
    """Barrel hinge: base knuckle(s) in the middle, lid ears at both ends.
    Axis at ZH and radius RB, so the same geometry also prints in place
    (lid opened 180 deg lies top-face-down next to the base)."""

    pip = False

    def __init__(self, xk=7.7, gax=0.3, g=0.3, pip=False, sweep=180, base_ranges=None, ear_x0=None, R=RB):
        self.R = R
        self.z_h = ZH
        self.y_h = Y_CLEAR + R
        self.y_top = self.y_h
        self.y_pm = self.y_h - R - 0.3
        self.xk, self.gax, self.g, self.pip, self.sweep = xk, gax, g, pip, sweep
        self.base_ranges = base_ranges or [(-xk, xk)]
        self.ear_x0 = ear_x0 if ear_x0 is not None else xk + gax

    def support_cs(self):
        y, z, R = self.y_h, self.z_h, self.R
        return rect(self.y_pm - 1.0, y, 0, T) + CS.batch_hull([circle(R, y, z), rect(y - 0.9 * R, y + 0.4 * R, 0, T)])

    def base(self):
        cs = self.support_cs()
        parts = [base_plate(self.y_pm), prism_x(rect(self.y_pm - 1.0, self.y_h, 0, T), -self.xk, self.xk)]
        parts += [prism_x(cs, x0, x1) for x0, x1 in self.base_ranges]
        return union(*parts)

    def lid(self, ears=True):
        lid = lid_plate(self.y_h)
        if ears:
            cs = circle(self.R, self.y_h, self.z_h)
            lid = union(lid, prism_x(cs, self.ear_x0, HW), prism_x(cs, -HW, -self.ear_x0))
        x_c = self.ear_x0 - 1e-3
        cut = prism_x(circle(self.R + self.g, self.y_h, self.z_h), -x_c, x_c)
        sw = sweep_cs(self.support_cs().offset(self.g, ROUND), self.y_h, self.z_h, 0, self.sweep)
        cut = cut + prism_x(sw, -x_c, x_c)
        return lid - cut

    def x_hole(self, r, x0, x1, tip):
        return prism_x(teardrop(r, self.y_h, self.z_h, tip), x0, x1)

    def motion(self):
        return {"type": "rotate", "y": self.y_h, "z": self.z_h, "max": 180}


class Posts:
    """Base carries two outer posts (bar / magnets), lid carries the middle element."""

    def __init__(self, xp, gax=0.4, g=0.3, sweep=180, R=RB):
        self.R, self.xp, self.gax, self.g, self.sweep = R, xp, gax, g, sweep
        self.z_h = ZH
        self.y_h = Y_CLEAR + R
        self.y_pm = self.y_h - R - 0.3

    def post_cs(self):
        y, z, R = self.y_h, self.z_h, self.R
        return rect(self.y_pm - 1.0, y, 0, T) + CS.batch_hull([circle(R, y, z), rect(y - 0.9 * R, y + 0.4 * R, 0, T)])

    def base(self):
        cs = self.post_cs()
        return union(base_plate(self.y_pm), prism_x(cs, self.xp, HW), prism_x(cs, -HW, -self.xp))

    def lid_cut(self):
        sw = sweep_cs(self.post_cs().offset(self.g, ROUND), self.y_h, self.z_h, 0, self.sweep)
        return prism_x(sw, self.xp - self.gax, HW + 1) + prism_x(sw, -HW - 1, -(self.xp - self.gax))

    def motion(self):
        return {"type": "rotate", "y": self.y_h, "z": self.z_h, "max": 180}


class Flex:
    """Flexure hinge. Geometry lives in the FLAT frame = print pose: the lid is
    opened 180 deg and lies top-face-down beside the frame; the strip lies on
    the bed between them. Closing bends the strip into a half circle."""

    CLOSED = 180.0

    def __init__(self, t, zm, y_p=12.0, xr=((-11.5, 11.5),)):
        self.t, self.zm, self.y_p = t, zm, y_p
        self.rho = (ZL1 - 2 * zm) / 2            # half-circle radius that lands the lid on the frame
        self.L = math.pi * self.rho
        self.xr = [list(r) for r in xr]
        self.F = (y_p + self.L, zm)

    def end_pose(self, alpha_deg):
        a = math.radians(alpha_deg)
        if abs(a) < 1e-9:
            return (self.y_p + self.L, self.zm)
        rho = self.L / a
        return (self.y_p + rho * math.sin(a), self.zm + rho * (1 - math.cos(a)))

    def lid_matrix(self, alpha_deg):
        ey, ez = self.end_pose(alpha_deg)
        return mat_translate(0, ey, ez) @ mat_rot_x(alpha_deg) @ mat_translate(0, -self.F[0], -self.F[1])

    def lid_flat(self):
        return apply(lid_plate(self.y_p), np.linalg.inv(self.lid_matrix(self.CLOSED)))

    def strip(self, overlap=0.4):
        return union(*[box(x0, x1, self.y_p - overlap, self.F[0] + overlap, self.zm - self.t / 2, self.zm + self.t / 2)
                       for x0, x1 in self.xr])

    def motion(self):
        return {"type": "flex", "yP": self.y_p, "zm": self.zm, "L": self.L, "t": self.t,
                "xr": self.xr, "Fy": self.F[0], "Fz": self.F[1], "closed": self.CLOSED}
