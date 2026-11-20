"""The 20 hinge variants of the FLAT 16 mm push-button cover (frame + flat lid).

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
from common import (HOLE_R, HW, T, ZH, ZL0, ZL1, Flex, Knuckle, Posts, P_lid_facedown, P_open,
                    base_plate, lid_plate)

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


def hw_aside(y_from, z_from, r_bottom, y_to=-24.0):
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
def _filament_pin(y, z):
    return union(xcyl(0.875, -12.55, 12.55, y, z),
                 xcone(1.25, 0.95, 12.5, 12.95, y, z), xcone(1.25, 0.95, -12.5, -12.95, y, z))


def _counterbores(lid, y, z):
    return lid - xcyl(1.35, HW - 0.5, HW + 1, y, z) - xcyl(1.35, -HW - 1, -HW + 0.5, y, z)


def v01():
    k = Knuckle()
    y, z = k.y_h, k.z_h
    base = k.base() - k.x_hole(0.925, -k.xk - 1, k.xk + 1, (0, 1))
    lid = _counterbores(k.lid() - k.x_hole(1.025, -HW - 1, HW + 1, (0, -1)), y, z)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
        part("filament-pin", _filament_pin(y, z), "hw-base", "hw", C_FIL, hw_aside(y, z, 1.25)),
    ], prints=[("base.stl", ["base"], "плоско, без поддержек"), ("lid.stl", ["lid"], "верхней стороной вниз")],
        motion=k.motion())


def v02():
    k = Knuckle()
    y, z = k.y_h, k.z_h
    base = k.base() - k.x_hole(0.725, -k.xk - 1, k.xk + 1, (0, 1))
    lid = k.lid() - k.x_hole(0.85, -HW - 1, HW + 1, (0, -1))
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
        part("steel-pin", xcyl(0.75, -12.85, 12.85, y, z), "hw-base", "hw", C_STEEL, hw_aside(y, z, 0.75)),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "верхней стороной вниз")],
        motion=k.motion())


def v03():
    k = Knuckle(xk=7.6, gax=0.4, g=0.35, pip=True)
    y, z = k.y_h, k.z_h
    top = 2.6  # slot roof: the pin is dropped in at the pause, then bridged over
    slot_b = CS.batch_hull([circle(0.8, y, z), rect(y - 0.8, y + 0.8, z, top)])
    slot_l = CS.batch_hull([circle(0.8, y, z), rect(y - 0.8, y + 0.8, 2 * z - top, z)])
    base = k.base() - prism_x(slot_b, -k.xk - 1, k.xk + 1)
    lid = k.lid() - prism_x(slot_l, -12.4, 12.4)
    pin = xcyl(0.75, -12.2, 12.2, y, z - 0.05)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_open(180, y, z)),
        part("steel-pin", pin, "hw-base", "hw", C_STEEL),
    ], prints=[("cover-print-in-place.stl", ["base", "lid"],
                "одним заданием; ПАУЗА перед слоем 2.6 мм - вложить стальной штифт 1.5 x 24.4")],
        motion=k.motion(), pause_z=2.6)


def v04():
    k = Knuckle(xk=9.5)
    y, z = k.y_h, k.z_h
    base = k.base() - xcyl(0.8, 6.5, 9.6, y, z) - xcyl(0.8, -9.6, -6.5, y, z)
    lid = k.lid() - xcyl(1.05, -HW - 1, -k.ear_x0 + 0.01, y, z) - xcyl(1.05, k.ear_x0 - 0.01, HW + 1, y, z)
    screws = union(xcyl(1.75, HW, HW + 1.3, y, z, 32), xcyl(1.0, 7.0, HW + 0.01, y, z, 24),
                   xcyl(1.75, -HW - 1.3, -HW, y, z, 32), xcyl(1.0, -HW - 0.01, -7.0, y, z, 24))
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
        part("screws-M2x6", screws, "hw-base", "hw", C_STEEL, hw_aside(y, z, 1.75)),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "верхней стороной вниз")],
        motion=k.motion())


# ======================================================================
# B. Printed parts snapped / slid together
# ======================================================================
def _lug_knuckle():
    return Knuckle(xk=HW - 1.2 - 0.25, gax=0.25, ear_x0=HW - 1.2)


def v05():
    k = _lug_knuckle()
    y, z = k.y_h, k.z_h
    xb, xe = k.xk, k.ear_x0
    base = k.base()
    for s in (1, -1):
        base = base - xcyl(0.7, s * (xb - 0.9), s * (xb + 1), y, z)
        base = base - M.batch_hull((xcyl(0.62, s * (xb - 0.75), s * (xb + 1), y, z),
                                    xcyl(0.62, s * (xb - 0.15), s * (xb + 1), y, z + k.R + 0.5)))
    lid = k.lid()
    for s in (1, -1):
        lid = lid + union(xcyl(0.6, s * (xe - 0.35), s * (xe + 0.05), y, z), xcone(0.6, 0.25, s * (xe - 0.35), s * (xe - 0.65), y, z))
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "верхней стороной вниз, PETG")],
        motion=k.motion())


def _hook_cs(y, z, r_in, r_out, center_deg, half_deg):
    a, h = math.radians(center_deg), math.radians(half_deg)
    wedge = CS.hull_points([(y, z)] + [(y + 5 * math.cos(a + t), z + 5 * math.sin(a + t)) for t in np.linspace(-h, h, 9)])
    return circle(r_out, y, z) - circle(r_in, y, z) - wedge


def _posts_hooks(r_in, center_deg, half_deg):
    p = Posts(xp=9.6)
    y, z = p.y_h, p.z_h
    base = union(p.base(), xcyl(0.75, -p.xp - 0.5, p.xp + 0.5, y, z))
    lid = lid_plate(y) - p.lid_cut() - xcyl(r_in, -p.xp, p.xp, y, z)
    cs = _hook_cs(y, z, r_in, p.R, center_deg, half_deg)
    x1 = p.xp - p.gax
    lid = union(lid, prism_x(cs, 2.5, x1), prism_x(cs, -x1, -2.5))
    return p, base, lid


def v06():
    p, base, lid = _posts_hooks(0.85, -45, 50)
    return dict(k=p, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "верхней стороной вниз")],
        motion=p.motion())


def v07():
    k = _lug_knuckle()
    y, z = k.y_h, k.z_h
    xb, xe = k.xk, k.ear_x0
    base = k.base()
    for s in (1, -1):
        base = base - xcone(1.05, 0.3, s * xb, s * (xb - 0.75), y, z) - xcyl(1.05, s * xb, s * (xb + 1), y, z)
        base = base - M.batch_hull((xcyl(0.5, s * (xb - 0.2), s * (xb + 1), y, z + k.R + 0.5),
                                    xcone(0.9, 0.4, s * xb, s * (xb - 0.5), y, z)))
    lid = k.lid()
    for s in (1, -1):
        lid = lid + xcone(0.9, 0.3, s * (xe + 0.05), s * (xe - 0.6), y, z)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "верхней стороной вниз, PETG")],
        motion=k.motion())


def v08():
    k = Knuckle(sweep=225)
    y, z = k.y_h, k.z_h
    slab = CS.square((5, 1.1)).translate((-2.5, -0.55)).rotate(50).translate((y, z))
    pin_cs = circle(0.8, y, z) ^ slab
    base = union(k.base(), prism_x(pin_cs, k.xk - 0.1, 9.4), prism_x(pin_cs, -9.4, -k.xk + 0.1))
    lid = k.lid()
    for s in (1, -1):
        x0, x1 = s * (k.ear_x0 - 0.01), s * 9.7
        lid = lid - xcyl(0.9, x0, x1, y, z) - prism_x(rect(y - 0.625, y + 0.625, z - 5, z), x0, x1)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "верхней стороной вниз")],
        motion=k.motion(), assembly_angle=220)


def v09():
    k = Knuckle()
    y, z = k.y_h, k.z_h
    flat = z - 0.6
    hole_b = teardrop(0.95, y, z, (0, 1)) ^ rect(y - 3, y + 3, flat - 0.05, z + 3)
    base = k.base() - prism_x(hole_b, -k.xk - 1, k.xk + 1)
    lid = k.lid() - k.x_hole(1.05, -HW - 1, HW + 1, (0, -1))
    pin = prism_x(circle(0.9, y, z) ^ rect(y - 3, y + 3, flat, z + 3), -12.95, 12.95)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
        part("d-pin", pin, "hw-base", "print", "#f2c14e", mat_translate(0, -24 - y, -flat)),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "верхней стороной вниз"),
               ("d-pin.stl", ["d-pin"], "лёжа на лыске (плоской гранью вниз)!")],
        motion=k.motion())


def v10():
    p, base, lid = _posts_hooks(0.9, 240, 55)
    return dict(k=p, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "верхней стороной вниз")],
        motion=p.motion())


# ======================================================================
# C. Print-in-place (lid opened 180 deg, lying top-face-down next to the frame)
# ======================================================================
def v11():
    k = Knuckle(xk=7.0, gax=0.4, g=0.35, pip=True)
    y, z = k.y_h, k.z_h
    base = union(k.base(), xcone(0.9, 0.25, k.xk - 0.01, k.xk + 0.65, y, z), xcone(0.9, 0.25, -k.xk + 0.01, -k.xk - 0.65, y, z))
    lid = k.lid()
    for s in (1, -1):
        lid = lid - xcone(1.05, 0.3, s * (k.ear_x0 - 0.01), s * (k.ear_x0 + 0.75), y, z)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_open(180, y, z)),
    ], prints=[("cover-print-in-place.stl", ["base", "lid"], "одним куском, раскрытой на 180°")],
        motion=k.motion())


def v12():
    k = Knuckle(xk=7.0, gax=0.4, g=0.35, pip=True)
    y, z = k.y_h, k.z_h
    pin_cs = CS.batch_hull([circle(0.55, y, z), CS.hull_points([(y, z - 0.78), (y - 0.2, z - 0.5), (y + 0.2, z - 0.5)])])
    base = union(k.base(), prism_x(pin_cs, -12.3, 12.3))
    hole = teardrop(1.05, y, z, (0, -1)) ^ rect(y - 3, y + 3, z - 1.3, z + 3)
    lid = k.lid()
    for s in (1, -1):
        lid = lid - prism_x(hole, s * (k.ear_x0 - 0.01), s * 12.45)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_open(180, y, z)),
    ], prints=[("cover-print-in-place.stl", ["base", "lid"], "одним куском, раскрытой на 180°")],
        motion=k.motion())


def v13():
    k = Knuckle(xk=7.6, gax=0.4, g=0.35, pip=True)
    y, z = k.y_h, k.z_h
    base = k.base() - k.x_hole(0.925, -k.xk - 1, k.xk + 1, (0, 1))
    lid = _counterbores(k.lid() - k.x_hole(1.025, -HW - 1, HW + 1, (0, -1)), y, z)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_open(180, y, z)),
        part("filament-pin", _filament_pin(y, z), "hw-base", "hw", C_FIL, hw_aside(y, z, 1.25, y_to=-22)),
    ], prints=[("cover-print-in-place.stl", ["base", "lid"], "одним куском на 180°, потом протолкнуть филамент")],
        motion=k.motion())


# ======================================================================
# D. Flexures (flat frame = print pose, strip bends 180 deg when closed)
# ======================================================================
def _flex(f, extra_parts, prints, color):
    return dict(k=f, flex=True, parts=[part("base", extra_parts.pop("base"), "base"),
                                        part("lid", extra_parts.pop("lid"), "lid")] + extra_parts.pop("more", []),
                prints=prints, motion=f.motion(), flex_color=color)


def v14():
    f = Flex(t=0.4, zm=0.2)
    d = dict(base=base_plate(f.y_p), lid=f.lid_flat(), more=[part("hinge", f.strip(), "flex", "print", C_LID)])
    return _flex(f, d, [("cover-living-hinge.stl", ["base", "hinge", "lid"], "одним куском, PP или PETG, линии поперёк сгиба")], C_LID)


def v15():
    f = Flex(t=0.3, zm=0.15, xr=((-11.5, -6.7), (-2.4, 2.4), (6.7, 11.5)))
    d = dict(base=base_plate(f.y_p), lid=f.lid_flat(), more=[part("hinge", f.strip(), "flex", "print", C_LID)])
    return _flex(f, d, [("cover-multiband-flexure.stl", ["base", "hinge", "lid"], "одним куском, PETG, слой 0.15")], C_LID)


def v16():
    f = Flex(t=0.8, zm=0.4)
    keep = zcyl(HOLE_R + 0.6, -1, 3)
    tab_b = box(-11.5, 11.5, f.y_p - 8, f.y_p + 0.01, 0, 0.8) - keep
    tab_l = box(-11.5, 11.5, f.F[0] - 0.01, f.F[0] + 4, 0, 0.8)
    holes_b = union(*[zcyl(0.6, -1, 2, sx * xx, yy) for sx in (-1, 1) for xx, yy in ((10.0, f.y_p - 5.5), (7.0, f.y_p - 2.5))],
                    zcyl(0.5, -1, 2, 0, f.y_p - 0.7))
    holes_l = union(*[zcyl(0.6, -1, 2, x, f.F[0] + 2) for x in (-6, 0, 6)])
    strap_b, strap_l = tab_b - holes_b, tab_l - holes_l
    d = dict(base=base_plate(f.y_p) - strap_b, lid=f.lid_flat() - strap_l,
             more=[part("tpu-anchor-base", strap_b, "base", "print", C_TPU),
                   part("tpu-anchor-lid", strap_l, "lid", "print", C_TPU)])
    return _flex(f, d, [("rigid-PETG.stl", ["base", "lid"], "мультиматериал: рамка и крышка PETG"),
                        ("hinge-TPU.stl", ["tpu-anchor-base", "tpu-anchor-lid", "@flex"], "мультиматериал: петля TPU 95A")], C_TPU)


def v17():
    f = Flex(t=0.2, zm=0.5, xr=((-12, 12),))
    fab_b = box(-12, 12, f.y_p - 9, f.y_p + 0.01, 0.4, 0.6) - zcyl(HOLE_R + 0.6, -1, 3)
    fab_l = box(-12, 12, f.F[0] - 0.01, f.F[0] + 5, 0.4, 0.6)
    d = dict(base=base_plate(f.y_p), lid=f.lid_flat(),
             more=[part("fabric-base", fab_b, "hw-base", "hw", C_FAB), part("fabric-lid", fab_l, "hw-lid", "hw", C_FAB)])
    r = _flex(f, d, [("cover-fabric.stl", ["base", "lid"], "одним заданием; ПАУЗА после слоя 0.4 мм - натянуть фатин/ленту")], C_FAB)
    r["pause_z"] = 0.4
    return r


def v18():
    f = Flex(t=0.12, zm=0.06, xr=((-11, 11),))
    pocket_b = box(-11.1, 11.1, f.y_p - 8, f.y_p + 1, -1, 0.15) - zcyl(HOLE_R + 0.5, -2, 3)
    pocket_l = box(-11.1, 11.1, f.F[0] - 1, f.F[0] + 5, -1, 0.15)
    film_b = box(-11, 11, f.y_p - 7.9, f.y_p + 0.01, 0, 0.12) - zcyl(HOLE_R + 0.6, -1, 3)
    film_l = box(-11, 11, f.F[0] - 0.01, f.F[0] + 4.9, 0, 0.12)
    d = dict(base=base_plate(f.y_p) - pocket_b, lid=f.lid_flat() - pocket_l,
             more=[part("film-base", film_b, "hw-base", "hw", C_FILM), part("film-lid", film_l, "hw-lid", "hw", C_FILM)])
    return _flex(f, d, [("cover-film.stl", ["base", "lid"], "одним заданием, две детали; плёнку вклеить в карманы")], C_FILM)


# ======================================================================
# E. Other
# ======================================================================
def v19():
    p = Posts(xp=10.3, gax=0.3)
    y, z = p.y_h, p.z_h
    x1 = p.xp - p.gax
    base = p.base() - xcyl(1.05, p.xp - 0.01, p.xp + 1.1, y, z) - xcyl(1.05, -p.xp - 1.1, -p.xp + 0.01, y, z)
    lid = union(lid_plate(y) - p.lid_cut(), xcyl(p.R, -x1, x1, y, z))
    lid = lid - xcyl(1.05, x1 - 1.1, x1 + 0.01, y, z) - xcyl(1.05, -x1 - 0.01, -x1 + 1.1, y, z)
    mags_b = union(xcyl(1.0, p.xp + 0.05, p.xp + 1.05, y, z), xcyl(1.0, -p.xp - 1.05, -p.xp - 0.05, y, z))
    mags_l = union(xcyl(1.0, x1 - 1.05, x1 - 0.05, y, z), xcyl(1.0, -x1 + 0.05, -x1 + 1.05, y, z))
    return dict(k=p, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
        part("magnets-base", mags_b, "hw-base", "hw", C_MAG, hw_aside(y, z, 1.0)),
        part("magnets-lid", mags_l, "hw-lid", "hw", "#c0392b", mat_translate(0, -4, 0) @ hw_aside(y, z, 1.0)),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "верхней стороной вниз")],
        motion=p.motion())


def v20():
    k = Knuckle(base_ranges=[(-7.7, -2.0), (2.0, 7.7)])
    y, z = k.y_h, k.z_h
    base = k.base() - k.x_hole(0.48, -k.xk - 1, k.xk + 1, (0, 1))
    base = base - xcyl(1.15, -1.95, 1.95, y, z) - box(-1.7, -1.1, y - 4.5, y, 0.55, T + 1)
    lid = k.lid() - k.x_hole(0.58, -HW - 1, HW + 1, (0, -1)) - box(1.1, 1.7, y - 3.8, y, 2.15, 2.75)
    rc, rw, turns, x0, x1 = 0.8, 0.15, 4.5, -1.4, 1.4
    path = [(x0, y - 4.2, z - rc), (x0, y - 0.9, z - rc)]
    nst = int(turns * 24)
    for i in range(nst + 1):
        a = -math.pi / 2 + 2 * math.pi * turns * i / nst
        path.append((x0 + (x1 - x0) * i / nst, y + rc * math.cos(a), z + rc * math.sin(a)))
    coil = tube(path, rw, 8)
    leg = tube([(x1, y, z + rc), (x1, y - 1.0, z + rc), (x1, y - 3.5, z + rc)], rw, 8)
    return dict(k=k, parts=[
        part("base", base, "base"),
        part("lid", lid, "lid", P=P_lid_facedown()),
        part("steel-pin", xcyl(0.5, -12.85, 12.85, y, z), "hw-base", "hw", C_STEEL, hw_aside(y, z, 0.5)),
        part("spring-coil", coil, "hw-base", "hw", "#9aa3ad", mat_translate(-16, -4, 0) @ hw_aside(y, z, 1.0)),
        part("spring-leg", leg, "hw-lid", "hw", "#9aa3ad", mat_translate(-16, -4, 0) @ hw_aside(y, z, 1.0)),
    ], prints=[("base.stl", ["base"], "плоско"), ("lid.stl", ["lid"], "верхней стороной вниз")],
        motion=k.motion())


ALL = [v01, v02, v03, v04, v05, v06, v07, v08, v09, v10,
       v11, v12, v13, v14, v15, v16, v17, v18, v19, v20]
