"""The 20 hinge variants of the 16 mm push-button flip cover.

Every variant returns a dict:
  parts:  list of {name, solid, group(base|lid|hw-base|hw-lid), role(print|hw), color, P (assembled->print 4x4)}
  prints: list of (file, [part names], note)   - downloadable print jobs (print pose)
  motion: kinematics for the viewer
"""
import math

import numpy as np
import manifold3d as m3

from geo import (CS, M, ROUND, apply, box, circle, mat_rot_x, mat_translate, prism_x, rect, teardrop,
                 union, xcone, xcyl, zcyl)
from common import (HOLE_R, HW, RIM, T, XIN, YB, ZF, Flex, Knuckle, Posts, P_lid_facedown, P_open,
                    base_plate, lid_shell, sweep_cs)

I4 = np.eye(4)
C_BASE = "#4a5463"
C_LID = "#cfe8ff"
C_STEEL = "#b8bec7"
C_FIL = "#ff8a3d"
C_TPU = "#2f9e6b"
C_MAG = "#8a8f98"
C_FAB = "#e85d75"
C_FILM = "#e3a32b"


def part(name, solid, group, role="print", color=None, P=I4):
    if color is None:
        color = C_BASE if group.endswith("base") else C_LID
    return {"name": name, "solid": solid, "group": group, "role": role, "color": color, "P": P}


def hw_aside(y_from, z_from, r_bottom, y_to=-22.0):
    """Display pose for a loose axle lying on the bed in front of the base."""
    return mat_translate(0, y_to - y_from, r_bottom - z_from)


def tube(path, r, seg=10):
    """Closed tube along a polyline path (for springs / wire legs)."""
    p = np.asarray(path, dtype=float)
    n = len(p)
    tang = np.gradient(p, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    ref = np.array([0.0, 0.0, 1.0]) if abs(tang[0][2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    nrm = np.cross(tang[0], ref); nrm /= np.linalg.norm(nrm)
    verts = []
    for i in range(n):
        if i > 0:
            nrm = nrm - tang[i] * np.dot(nrm, tang[i])
            nrm /= np.linalg.norm(nrm)
        b = np.cross(tang[i], nrm)
        for k in range(seg):
            a = 2 * math.pi * k / seg
            verts.append(p[i] + r * (math.cos(a) * nrm + math.sin(a) * b))
    verts.append(p[0]); verts.append(p[-1])
    c0, c1 = len(verts) - 2, len(verts) - 1
    tris = []
    for i in range(n - 1):
        for k in range(seg):
            a, b_ = i * seg + k, i * seg + (k + 1) % seg
            c, d = a + seg, b_ + seg
            tris += [[a, c, b_], [b_, c, d]]
    for k in range(seg):
        tris.append([c0, k, (k + 1) % seg])
        tris.append([c1, (n - 1) * seg + (k + 1) % seg, (n - 1) * seg + k])
    tris = np.asarray(tris, dtype=np.uint32)
    solid = M(m3.Mesh(vert_properties=np.asarray(verts, dtype=np.float32), tri_verts=tris))
    if solid.is_empty() or solid.volume() <= 0:
        solid = M(m3.Mesh(vert_properties=np.asarray(verts, dtype=np.float32), tri_verts=tris[:, ::-1].copy()))
    assert not solid.is_empty(), "tube is not manifold"
    return solid


# ======================================================================
# A. Separate (non-printed) axle
# ======================================================================
def v01():
    k = Knuckle(R=2.0)
    y, z = k.y_h, k.z_h
    base = k.base() - k.x_hole(0.925, -k.xk - 1, k.xk + 1, (0, 1))
    lid = k.lid() - k.x_hole(1.025, -HW - 1, HW + 1, (0, -1))
    lid = lid - xcyl(1.45, HW - 0.6, HW + 1, y, z) - xcyl(1.45, -HW - 1, -HW + 0.6, y, z)
    pin = union(xcyl(0.875, -11.45, 11.45, y, z),
                xcone(1.35, 1.0, 11.4, 11.95, y, z), xcone(1.35, 1.0, -11.4, -11.95, y, z))
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
        part("filament-pin", pin, "hw-base", "hw", C_FIL, hw_aside(y, z, 1.35)),
    ], prints=[("base.stl", ["base"], "PETG/PLA, плоско, без поддержек"),
               ("lid.stl", ["lid"], "лицом вниз, без поддержек")],
        motion=k.motion(95))


def v02():
    k = Knuckle(R=1.7)
    y, z = k.y_h, k.z_h
    base = k.base() - k.x_hole(0.725, -k.xk - 1, k.xk + 1, (0, 1))
    lid = k.lid() - k.x_hole(0.85, -HW - 1, HW + 1, (0, -1))
    pin = xcyl(0.75, -11.85, 11.85, y, z)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
        part("steel-pin", pin, "hw-base", "hw", C_STEEL, hw_aside(y, z, 0.75)),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "лицом вниз")],
        motion=k.motion(95))


def v03():
    k = Knuckle(R=2.2, xk=7.6, gax=0.4, g=0.4, pip=True, sweep=180)
    y, z = k.y_h, k.z_h
    base = k.base() - k.x_hole(0.8, -k.xk - 1, k.xk + 1, (0, 1))
    lid = k.lid() - k.x_hole(0.95, -11.4, 11.4, (0, -1))
    pin = xcyl(0.75, -11.25, 11.25, y, z)
    P = P_open(180, y, z)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P),
        part("steel-pin", pin, "hw-base", "hw", C_STEEL),
    ], prints=[("cover-print-in-place.stl", ["base", "lid"],
                "одним заданием; ПАУЗА перед слоем 4.6 мм (после 4.4) - вложить стальной штифт 1.5x22.5")],
        motion=k.motion(95), pause_z=4.6)


def v04():
    k = Knuckle(R=2.8, xk=7.6)
    y, z = k.y_h, k.z_h
    base = k.base() - k.x_hole(1.125, -k.xk - 1, k.xk + 1, (0, 1))
    lid = k.lid() - k.x_hole(1.1, -HW - 1, HW + 1, (0, -1))
    lid = lid - xcyl(2.0, 10.6, HW + 1, y, z)                                   # head counterbore
    hexcs = CS.circle(4.1 / math.sqrt(3), 6).rotate(90).translate((y, z))      # M2 nut pocket
    lid = lid - prism_x(hexcs, -HW - 1, -10.3)
    screw = union(xcyl(1.9, 10.7, 12.0, y, z, 32), xcyl(1.0, -11.9, 10.71, y, z, 24))
    nut = prism_x(CS.circle(4.0 / math.sqrt(3), 6).rotate(90).translate((y, z)), -11.95, -10.35) - xcyl(1.0, -13, -9, y, z)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
        part("screw-M2x22", screw, "hw-lid", "hw", C_STEEL, hw_aside(y, z, 1.9)),
        part("nut-M2", nut, "hw-lid", "hw", C_STEEL, mat_translate(18, 0, 0) @ hw_aside(y, z, 2.0)),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "лицом вниз")],
        motion=k.motion(95))


# ======================================================================
# B. Printed parts snapped / slid together
# ======================================================================
def _wall_knuckle(R=2.0):
    return Knuckle(R=R, xk=XIN - 0.25, gax=0.25, ear_x0=XIN)


def v05():
    k = _wall_knuckle()
    y, z = k.y_h, k.z_h
    xb = k.xk
    base = k.base()
    for s in (1, -1):
        base = base - xcyl(0.9, s * (xb - 1.1), s * (xb + 1), y, z)
        ramp = M.batch_hull((xcyl(0.85, s * (xb - 0.95), s * (xb + 1), y, z),
                            xcyl(0.85, s * (xb - 0.15), s * (xb + 1), y, z + k.R + 0.6)))
        base = base - ramp
    lid = k.lid()
    for s in (1, -1):
        lid = lid + union(xcyl(0.8, s * (XIN - 0.55), s * (XIN + 0.05), y, z),
                          xcone(0.8, 0.35, s * (XIN - 0.55), s * (XIN - 0.95), y, z))
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "лицом вниз, PETG")],
        motion=k.motion(95))


def v06():
    p = Posts(Rp=2.2, xp=8.4, y_h=12.15)
    y, z = p.y_h, p.z_h
    r_bar, r_in, r_out = 1.1, 1.25, 2.25
    y_top = y + r_out
    base = union(p.base(), xcyl(r_bar, -p.xp - 0.5, p.xp + 0.5, y, z))
    lid = lid_shell(y_top) - p.lid_cut() - xcyl(r_in, -p.xp, p.xp, y, z)
    dirv = (-1 / math.sqrt(2), -1 / math.sqrt(2))
    hook_cs = (circle(r_out, y, z) + rect(y, y_top, z, ZF)) - circle(r_in, y, z)
    a = math.atan2(dirv[1], dirv[0]); h = math.radians(50)
    wedge = CS.hull_points([(y, z), (y + 6 * math.cos(a - h), z + 6 * math.sin(a - h)),
                            (y + 6 * math.cos(a), z + 6 * math.sin(a)),
                            (y + 6 * math.cos(a + h), z + 6 * math.sin(a + h))])
    hook_cs = hook_cs - wedge
    x1 = p.xp - p.gax
    lid = union(lid, prism_x(hook_cs, 3.2, x1), prism_x(hook_cs, -x1, -3.2))
    return dict(k=p, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "лицом вниз")],
        motion=p.motion(95))


def v07():
    k = _wall_knuckle()
    y, z = k.y_h, k.z_h
    xb = k.xk
    base = k.base()
    for s in (1, -1):
        sock = xcone(1.45, 0.25, s * xb, s * (xb - 1.2), y, z)
        base = base - sock - xcyl(1.45, s * xb, s * (xb + 1), y, z)
        ramp = M.batch_hull((xcyl(0.7, s * (xb - 0.2), s * (xb + 1), y, z + k.R + 0.6),
                            xcone(1.2, 0.5, s * xb, s * (xb - 0.7), y, z)))
        base = base - ramp
    lid = k.lid()
    for s in (1, -1):
        lid = lid + xcone(1.2, 0.25, s * (XIN + 0.05), s * (XIN - 0.95), y, z)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "лицом вниз, PETG")],
        motion=k.motion(95))


def v08():
    k = Knuckle(R=2.0, sweep=140)
    y, z = k.y_h, k.z_h
    slab = CS.square((5, 1.4)).translate((-2.5, -0.7)).rotate(-45).translate((y, z))
    pin_cs = circle(1.0, y, z) ^ slab
    base = union(k.base(), prism_x(pin_cs, k.xk - 0.1, 9.6), prism_x(pin_cs, -9.6, -k.xk + 0.1))
    lid = k.lid()
    for s in (1, -1):
        x0, x1 = sorted((s * k.ear_x0 - s * 0.01, s * 9.9))
        lid = lid - xcyl(1.12, x0, x1, y, z) - prism_x(rect(y - 0.8, y + 0.8, z - 6, z), x0, x1)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "лицом вниз")],
        motion=k.motion(95), assembly_angle=135)


def v09():
    k = Knuckle(R=2.3)
    y, z = k.y_h, k.z_h
    flat = z - 0.8
    hole_b = teardrop(1.2, y, z, (0, 1)) ^ rect(y - 3, y + 3, flat - 0.05, z + 3)
    base = k.base() - prism_x(hole_b, -k.xk - 1, k.xk + 1)
    lid = k.lid() - k.x_hole(1.3, -HW - 1, HW + 1, (0, -1)) - xcyl(1.95, HW - 0.7, HW + 1, y, z)
    d_cs = circle(1.15, y, z) ^ rect(y - 3, y + 3, flat, z + 3)
    head_cs = circle(1.8, y, z) ^ rect(y - 3, y + 3, flat, z + 3)
    pin = union(prism_x(d_cs, -11.95, 11.32), prism_x(head_cs, 11.3, 11.95))
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
        part("d-pin", pin, "hw-base", "print", "#f2c14e", mat_translate(0, -22 - y, -flat)),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "лицом вниз"),
               ("d-pin.stl", ["d-pin"], "лёжа на лыске (плоской гранью вниз)!")],
        motion=k.motion(95))


def v10():
    p = Posts(Rp=2.2, xp=8.4, y_h=12.1)
    y, z = p.y_h, p.z_h
    r_bar, r_in, r_out = 1.1, 1.3, 2.3
    y_top = y + r_out
    base = union(p.base(), xcyl(r_bar, -p.xp - 0.5, p.xp + 0.5, y, z))
    lid = lid_shell(y_top) - p.lid_cut() - xcyl(r_in, -p.xp, p.xp, y, z)
    hook_cs = (circle(r_out, y, z) + rect(y, y_top, z, ZF)) - circle(r_in, y, z)
    a = math.radians(-100); h = math.radians(75)
    wedge = CS.hull_points([(y, z)] + [(y + 6 * math.cos(a + t), z + 6 * math.sin(a + t))
                                       for t in np.linspace(-h, h, 7)])
    hook_cs = hook_cs - wedge
    x1 = p.xp - p.gax
    lid = union(lid, prism_x(hook_cs, 3.2, x1), prism_x(hook_cs, -x1, -3.2))
    return dict(k=p, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "лицом вниз")],
        motion=p.motion(95))


# ======================================================================
# C. Print-in-place
# ======================================================================
def v11():
    k = Knuckle(R=3.0, xk=7.0, gax=0.4, g=0.4, pip=True, sweep=180)
    y, z = k.y_h, k.z_h
    base = union(k.base(), xcone(1.8, 0.4, k.xk - 0.01, k.xk + 1.4, y, z), xcone(1.8, 0.4, -k.xk + 0.01, -k.xk - 1.4, y, z))
    lid = k.lid()
    for s in (1, -1):
        lid = lid - xcone(1.9, 0.4, s * (k.ear_x0 - 0.01), s * (k.ear_x0 + 1.5), y, z)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_open(180, y, z)),
    ], prints=[("cover-print-in-place.stl", ["base", "lid"], "одним куском, раскрытой на 180°")],
        motion=k.motion(95))


def v12():
    k = Knuckle(R=3.0, xk=7.0, gax=0.4, g=0.4, pip=True, sweep=180)
    y, z = k.y_h, k.z_h
    pin_cs = CS.batch_hull([circle(1.2, y, z), CS.hull_points([(y, z - 1.7), (y - 0.3, z - 1.3), (y + 0.3, z - 1.3)])])
    base = union(k.base(), prism_x(pin_cs, -11.2, 11.2))
    hole = teardrop(2.05, y, z, (0, -1)) ^ rect(y - 4, y + 4, z - 2.5, z + 4)
    lid = k.lid()
    for s in (1, -1):
        x0, x1 = sorted((s * (k.ear_x0 - 0.01), s * 11.4))
        lid = lid - prism_x(hole, x0, x1)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_open(180, y, z)),
    ], prints=[("cover-print-in-place.stl", ["base", "lid"], "одним куском, раскрытой на 180°")],
        motion=k.motion(95))


def v13():
    k = Knuckle(R=2.2, xk=7.6, gax=0.4, g=0.4, pip=True, sweep=180)
    y, z = k.y_h, k.z_h
    base = k.base() - k.x_hole(0.925, -k.xk - 1, k.xk + 1, (0, 1))
    lid = k.lid() - k.x_hole(1.025, -HW - 1, HW + 1, (0, -1))
    lid = lid - xcyl(1.45, HW - 0.6, HW + 1, y, z) - xcyl(1.45, -HW - 1, -HW + 0.6, y, z)
    pin = union(xcyl(0.875, -11.45, 11.45, y, z),
                xcone(1.35, 1.0, 11.4, 11.95, y, z), xcone(1.35, 1.0, -11.4, -11.95, y, z))
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_open(180, y, z)),
        part("filament-pin", pin, "hw-base", "hw", C_FIL, P_open(0, y, z) @ hw_aside(y, z, 1.35, y_to=-20)),
    ], prints=[("cover-print-in-place.stl", ["base", "lid"], "одним куском на 180°, потом вставить филамент")],
        motion=k.motion(95))


# ======================================================================
# D. Flexures (flat frame = print pose)
# ======================================================================
def _flex_common(f, onepiece=True):
    base = base_plate(f.y_p)
    lid = f.lid_flat()
    return base, lid


def v14():
    f = Flex(L=3.0, t=0.4, zm=0.2)
    base, lid = _flex_common(f)
    strip = f.strip()
    return dict(k=f, flex=True, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid"),
        part("hinge", strip, "flex", "print", C_LID),
    ], prints=[("cover-living-hinge.stl", ["base", "hinge", "lid"], "одним куском, PP или PETG, линии поперёк сгиба")],
        motion=f.motion(), flex_color=C_LID)


def v15():
    xr = ((-10.6, -6.4), (-2.1, 2.1), (6.4, 10.6))
    f = Flex(L=4.0, t=0.3, zm=0.15, xr=xr)
    base, lid = _flex_common(f)
    strip = f.strip()
    return dict(k=f, flex=True, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid"),
        part("hinge", strip, "flex", "print", C_LID),
    ], prints=[("cover-multiband-flexure.stl", ["base", "hinge", "lid"], "одним куском, PETG, слой 0.15")],
        motion=f.motion(), flex_color=C_LID)


def v16():
    f = Flex(L=2.5, t=0.8, zm=0.4, xr=((-10.6, 10.6),))
    base, lid = _flex_common(f)
    keep_hole = zcyl(HOLE_R + 0.6, -1, 3)
    tab_b = box(-10.6, 10.6, f.y_p - 8, f.y_p + 0.01, 0, 0.8) - keep_hole      # U-shaped anchor around the button
    tab_l = box(-10.6, 10.6, f.F[0] - 0.01, f.F[0] + 4, 0, 0.8)
    holes_b = union(*[zcyl(0.8, -1, 2, sx * 9.6, f.y_p - dy) for sx in (-1, 1) for dy in (3.0, 6.5)])
    holes_l = union(*[zcyl(0.8, -1, 2, x, f.F[0] + 2) for x in (-6, 0, 6)])
    base = base - (tab_b - holes_b)
    lid = lid - (tab_l - holes_l)
    strap_b = tab_b - holes_b
    strap_l = tab_l - holes_l
    return dict(k=f, flex=True, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid"),
        part("tpu-anchor-base", strap_b, "base", "print", C_TPU),
        part("tpu-anchor-lid", strap_l, "lid", "print", C_TPU),
    ], prints=[("rigid-PETG.stl", ["base", "lid"], "мультиматериал: тело PETG"),
               ("hinge-TPU.stl", ["tpu-anchor-base", "tpu-anchor-lid", "@flex"], "мультиматериал: петля TPU 95A")],
        motion=f.motion(), flex_color=C_TPU)


def v17():
    f = Flex(L=2.0, t=0.2, zm=0.5, xr=((-11, 11),))
    base, lid = _flex_common(f)
    fab_b = box(-11, 11, f.y_p - 9, f.y_p + 0.01, 0.4, 0.6) - zcyl(HOLE_R + 0.6, -1, 3)
    fab_l = box(-11, 11, f.F[0] - 0.01, f.F[0] + 4.5, 0.4, 0.6)
    return dict(k=f, flex=True, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid"),
        part("fabric-base", fab_b, "hw-base", "hw", C_FAB),
        part("fabric-lid", fab_l, "hw-lid", "hw", C_FAB),
    ], prints=[("cover-fabric.stl", ["base", "lid"], "одним заданием; ПАУЗА после слоя 0.4 мм - натянуть фатин/ленту")],
        motion=f.motion(), flex_color=C_FAB, pause_z=0.4)


def v18():
    f = Flex(L=2.0, t=0.12, zm=0.06, y_p=9.7, xr=((-10, 10),))
    base, lid = _flex_common(f)
    pocket_b = box(-10.1, 10.1, f.y_p - 8, f.y_p + 1, -1, 0.15) - zcyl(HOLE_R + 0.5, -2, 3)
    pocket_l = box(-10.1, 10.1, f.F[0] - 1, f.F[0] + 4.5, -1, 0.15)
    base = base - pocket_b
    lid = lid - pocket_l
    film_b = box(-10, 10, f.y_p - 7.9, f.y_p + 0.01, 0, 0.12) - zcyl(HOLE_R + 0.6, -1, 3)
    film_l = box(-10, 10, f.F[0] - 0.01, f.F[0] + 4.4, 0, 0.12)
    return dict(k=f, flex=True, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid"),
        part("film-base", film_b, "hw-base", "hw", C_FILM),
        part("film-lid", film_l, "hw-lid", "hw", C_FILM),
    ], prints=[("cover-film.stl", ["base", "lid"], "одним заданием, две детали; плёнку вклеить в карманы")],
        motion=f.motion(), flex_color=C_FILM)


# ======================================================================
# E. Other
# ======================================================================
def v19():
    p = Posts(Rp=2.2, xp=9.2, gax=0.3)
    y, z = p.y_h, p.z_h
    Rl = 2.2
    y_top = y + Rl
    base = p.base()
    for s in (1, -1):
        x0, x1 = sorted((s * (p.xp - 0.01), s * (p.xp + 2.1)))
        base = base - xcyl(1.55, x0, x1, y, z)
    lid = lid_shell(y_top) - p.lid_cut()
    kn_cs = circle(Rl, y, z) + rect(y - Rl, y_top, z, ZF)
    x1 = p.xp - p.gax
    kn = prism_x(kn_cs, -x1, x1)
    lid = union(lid, kn)
    for s in (1, -1):
        a0, a1 = sorted((s * (x1 + 0.01), s * (x1 - 2.1)))
        lid = lid - xcyl(1.55, a0, a1, y, z)
    mags_b = union(xcyl(1.5, p.xp + 0.05, p.xp + 2.05, y, z), xcyl(1.5, -p.xp - 2.05, -p.xp - 0.05, y, z))
    mags_l = union(xcyl(1.5, x1 - 2.05, x1 - 0.05, y, z), xcyl(1.5, -x1 + 0.05, -x1 + 2.05, y, z))
    return dict(k=p, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
        part("magnets-base", mags_b, "hw-base", "hw", C_MAG, hw_aside(y, z, 1.5)),
        part("magnets-lid", mags_l, "hw-lid", "hw", "#c0392b", mat_translate(0, -5, 0) @ hw_aside(y, z, 1.5)),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "лицом вниз")],
        motion=p.motion(95))


def v20():
    k = Knuckle(R=2.0, base_ranges=[(-7.7, -2.2), (2.2, 7.7)])
    y, z = k.y_h, k.z_h
    base = k.base() - k.x_hole(0.725, -k.xk - 1, k.xk + 1, (0, 1))
    # groove for the base leg of the spring
    base = base - box(-1.75, -1.05, y - 6.2, y, T - 0.3, T + 1)
    lid = k.lid() - k.x_hole(0.85, -HW - 1, HW + 1, (0, -1))
    pin = xcyl(0.75, -11.85, 11.85, y, z)
    rc, rw = 1.15, 0.2
    turns, x0, x1 = 5.5, -1.35, 1.35
    # base leg lying in the groove, then the coil (starts at the bottom, ends at the top)
    path = [(-1.4, y - 6.0, T - 0.05), (-1.4, y - 2.2, T - 0.05), (-1.4, y - 0.6, z - rc)]
    nst = int(turns * 24)
    for i in range(1, nst + 1):
        a = -math.pi / 2 + 2 * math.pi * turns * i / nst
        xx = x0 + (x1 - x0) * i / nst
        path.append((xx, y + rc * math.cos(a), z + rc * math.sin(a)))
    coil = tube(path, rw, 8)
    # lid leg: from the top of the coil straight out, hooked into the lid top wall
    leg_l = tube([(x1, y, z + rc), (x1, y + 0.05, z + 1.6), (x1, y + 0.05, z + 2.9), (x1, y + 1.5, z + 3.0)], rw, 8)
    lid = lid - prism_x(circle(0.3, y + 1.4, z + 3.0), x1 - 0.6, x1 + 0.6)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
        part("steel-pin", pin, "hw-base", "hw", C_STEEL, hw_aside(y, z, 0.75)),
        part("spring-coil", coil, "hw-base", "hw", "#9aa3ad", mat_translate(-15, -4, 0) @ hw_aside(y, z, 1.4)),
        part("spring-leg", leg_l, "hw-lid", "hw", "#9aa3ad", mat_translate(-15, -4, 0) @ hw_aside(y, z, 1.4)),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "лицом вниз")],
        motion=k.motion(95))


ALL = [v01, v02, v03, v04, v05, v06, v07, v08, v09, v10,
       v11, v12, v13, v14, v15, v16, v17, v18, v19, v20]
