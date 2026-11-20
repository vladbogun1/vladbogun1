"""Build all 20 variants: viewer meshes, print-ready STLs, collision report, data.js.

usage: python3 build.py [ids...]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from geo import apply, box, mat_rot_x, mesh_payload, union, write_stl  # noqa: E402
from common import button_dummy  # noqa: E402
import variants  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "site"))
MODELS = os.path.join(ROOT, "models")
MESH = os.path.join(ROOT, "mesh")
PANEL = box(-40, 40, -40, 40, -5, 0)


def vol(a):
    return 0.0 if a.is_empty() else a.volume()


def collisions(v):
    parts = v["parts"]
    base = [p["solid"] for p in parts if p["group"] in ("base", "hw-base") and p["name"] != "hinge"]
    lid = [p["solid"] for p in parts if p["group"] in ("lid", "hw-lid")]
    lid_rigid = [p["solid"] for p in parts if p["group"] == "lid"]
    base_rigid = [p["solid"] for p in parts if p["group"] == "base"]
    B, L = union(*base), union(*lid)
    BR, LR = union(*base_rigid), union(*lid_rigid)
    hwb = [p["solid"] for p in parts if p["group"] == "hw-base"]
    BH = union(*hwb) if hwb else None
    rep = {"max_overlap": 0.0, "at": None, "panel_free_to": None}
    mo = v["motion"]
    if mo["type"] == "rotate":
        angles = list(range(0, max(180, v.get("assembly_angle") or 0) + 1, 5))
        for a in angles:
            Mx = mat_rot_x(-a, mo["y"], mo["z"])
            # hardware-vs-hardware is skipped: e.g. a spring is one wire split into a coil and a moving leg
            o = vol(apply(L, Mx) ^ BR) + (vol(apply(LR, Mx) ^ BH) if BH is not None else 0.0)
            if o > rep["max_overlap"]:
                rep["max_overlap"], rep["at"] = o, a
        free = 0
        for a in range(0, 181, 1):
            if vol(apply(union(*lid_rigid), mat_rot_x(-a, mo["y"], mo["z"])) ^ PANEL) > 1e-3:
                break
            free = a
        rep["panel_free_to"] = free
    else:
        f = v["k"]
        for al in range(0, 181, 5):
            Lr = apply(L, f.lid_matrix(al))
            o = vol(Lr ^ B)
            if o > rep["max_overlap"]:
                rep["max_overlap"], rep["at"] = o, 180 - al
        rep["panel_free_to"] = 180
    # rigid-vs-rigid in closed pose (ignores intentional press fits of hardware)
    if mo["type"] == "rotate":
        rep["closed_rigid_overlap"] = vol(union(*lid_rigid) ^ union(*base_rigid))
    return rep


def mat_list(m):
    return [float(x) for x in np.asarray(m).T.reshape(-1)]  # column-major for three.js


def build(idx, fn):
    t0 = time.time()
    v = fn()
    vid = f"v{idx:02d}"
    out = os.path.join(MODELS, vid)
    os.makedirs(os.path.join(out, "print"), exist_ok=True)
    meta_parts = []
    bundle = {}
    for p in v["parts"]:
        if p["name"] == "hinge":
            continue  # bending strip is generated live by the viewer
        fname = f"{p['name']}.stl"
        ntri = write_stl(p["solid"], os.path.join(out, fname))
        bundle[p["name"]] = mesh_payload(p["solid"])
        meta_parts.append({"name": p["name"], "file": f"models/{vid}/{fname}", "group": p["group"],
                           "role": p["role"], "color": p["color"], "P": mat_list(p["P"]), "tris": ntri})
    with open(os.path.join(MESH, f"{vid}.json"), "w") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    by_name = {p["name"]: p for p in v["parts"]}
    prints = []
    for fname, names, note in v["prints"]:
        solids = []
        for n in names:
            if n == "@flex":
                solids.append(v["k"].strip())
                continue
            p = by_name[n]
            solids.append(apply(p["solid"], p["P"]))
        s = union(*solids)
        bb = s.bounding_box()
        # drop onto the bed
        if abs(bb[2]) > 1e-6:
            s = s.translate((0, 0, -bb[2]))
        path = os.path.join(out, "print", fname)
        write_stl(s, path)
        bb = s.bounding_box()
        prints.append({"file": f"models/{vid}/print/{fname}", "name": fname, "note": note,
                       "size": [round(bb[3] - bb[0], 1), round(bb[4] - bb[1], 1), round(bb[5] - bb[2], 1)],
                       "volume_mm3": round(s.volume(), 1), "kb": round(os.path.getsize(path) / 1024)})
    rep = collisions(v)
    mo = dict(v["motion"])
    if mo["type"] == "rotate":
        mo["max"] = min(rep["panel_free_to"], 180)
    rigid = [p for p in v["parts"] if p["group"] in ("base", "lid") and p["name"] != "hinge"]
    if mo["type"] == "flex":
        closed = union(*[apply(p["solid"], v["k"].lid_matrix(v["k"].CLOSED)) if p["group"] == "lid" else p["solid"] for p in rigid])
    else:
        closed = union(*[p["solid"] for p in rigid])
    bb = closed.bounding_box()
    dims = [round(bb[3] - bb[0], 1), round(bb[4] - bb[1], 1), round(bb[5] - max(bb[2], 0), 1)]
    meta = {"id": idx, "dims": dims, "vid": vid, "parts": meta_parts, "prints": prints, "motion": mo,
            "flexColor": v.get("flex_color"), "pauseZ": v.get("pause_z"), "assemblyAngle": v.get("assembly_angle"),
            "check": rep}
    print(f"{vid}: overlap={rep['max_overlap']:.4f}@{rep['at']} closed={rep.get('closed_rigid_overlap', 0):.4f} "
          f"panel_free={rep['panel_free_to']} prints={[p['size'] for p in prints]} ({time.time() - t0:.1f}s)", flush=True)
    return meta


def main():
    os.makedirs(MODELS, exist_ok=True)
    ids = [int(a) for a in sys.argv[1:]] or list(range(1, 21))
    panel, btn, ring = button_dummy()
    common = os.path.join(MODELS, "common")
    os.makedirs(common, exist_ok=True)
    write_stl(panel, os.path.join(common, "panel.stl"))
    write_stl(btn, os.path.join(common, "button.stl"))
    write_stl(ring, os.path.join(common, "button-ring.stl"))
    os.makedirs(MESH, exist_ok=True)
    with open(os.path.join(MESH, "common.json"), "w") as fh:
        json.dump({"panel": mesh_payload(panel), "button": mesh_payload(btn), "button-ring": mesh_payload(ring)}, fh,
                  separators=(",", ":"))
    data_path = os.path.join(ROOT, "models.json")
    data = {}
    if os.path.exists(data_path):
        data = {d["id"]: d for d in json.load(open(data_path))}
    for i in ids:
        data[i] = build(i, variants.ALL[i - 1])
    out = [data[k] for k in sorted(data)]
    json.dump(out, open(data_path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
