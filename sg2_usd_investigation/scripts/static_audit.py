#!/usr/bin/env python3
"""Read SG2 source assets; write only audit artifacts under the investigation folder.

Run with Isaac Sim's Python after sourcing the prior investigation's env.sh.
No SimulationApp, physics step, source asset edits, or network requests are used.
The default candidate is the prior v4 USD, not an assertion of hardware accuracy.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pxr

if os.environ.get("PHYSX_P"):
    pxr.__path__.append(os.environ["PHYSX_P"] + "/pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT.parent / "sg2_usd_investigation"
TOLERANCE = {"mass_kg_atol": 1e-5, "com_m_atol": 2e-7,
             "tensor_kg_m2_atol": 2e-6, "tensor_rtol": 2e-5,
             "world_com_m_atol": 2e-7}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value, default="0 0 0"):
    return np.array([float(x) for x in (value or default).split()], dtype=float)


def rpy_rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin_matrix(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, :3] = rpy_rotation(vector(element.get("rpy")))
        transform[:3, 3] = vector(element.get("xyz"))
    return transform


def load_urdf(path):
    root = ET.parse(path).getroot()
    links, joints = {}, {}
    for link in root.findall("link"):
        name = link.get("name")
        if name in links:
            raise ValueError(f"Duplicate URDF link: {name}")
        inertial = link.find("inertial")
        row = {"name": name, "has_inertial": inertial is not None}
        if inertial is not None:
            origin = origin_matrix(inertial.find("origin"))
            inertia = inertial.find("inertia")
            get = lambda key: float(inertia.get(key, "0"))
            tensor = np.array([[get("ixx"), get("ixy"), get("ixz")],
                               [get("ixy"), get("iyy"), get("iyz")],
                               [get("ixz"), get("iyz"), get("izz")]])
            row.update(mass_kg=float(inertial.find("mass").get("value")),
                       com_link_m=origin[:3, 3], inertial_rotation=origin[:3, :3],
                       inertia_inertial_kg_m2=tensor,
                       inertia_link_kg_m2=origin[:3, :3] @ tensor @ origin[:3, :3].T)
        links[name] = row
    children = set()
    for joint in root.findall("joint"):
        name, child, parent = joint.get("name"), joint.find("child").get("link"), joint.find("parent").get("link")
        children.add(child)
        joints[name] = {"type": joint.get("type"), "parent": parent, "child": child,
                        "origin_matrix": origin_matrix(joint.find("origin")),
                        "limit": joint.find("limit").attrib if joint.find("limit") is not None else {},
                        "mimic": joint.find("mimic").attrib if joint.find("mimic") is not None else None}
    roots = set(links) - children
    transforms = {name: np.eye(4) for name in roots}
    pending = list(joints.values())
    while pending:
        progress = False
        for joint in list(pending):
            if joint["parent"] in transforms:
                # q=0 for every joint, including mimic; SG2 mimic offsets are zero.
                mimic = joint["mimic"]
                if mimic is not None and float(mimic.get("offset", "0")) != 0:
                    raise ValueError("Nonzero mimic offset needs explicit FK handling")
                transforms[joint["child"]] = transforms[joint["parent"]] @ joint["origin_matrix"]
                pending.remove(joint)
                progress = True
        if not progress:
            raise ValueError("URDF FK has missing parents or a cycle")
    total, weighted = 0.0, np.zeros(3)
    for name, row in links.items():
        row["world_matrix_zero_q"] = transforms[name]
        if row["has_inertial"]:
            row["com_world_zero_q_m"] = (transforms[name] @ np.r_[row["com_link_m"], 1])[:3]
            total += row["mass_kg"]
            weighted += row["mass_kg"] * row["com_world_zero_q_m"]
    return {"name": root.get("name"), "root_links": sorted(roots), "links": links, "joints": joints,
            "total_mass_kg": total, "whole_robot_com_zero_q_m": weighted / total,
            "gazebo_sensor_declarations": [{"reference": g.get("reference"), "name": sensor.get("name"),
                                             "type": sensor.get("type")}
                                            for g in root.findall("gazebo") for sensor in g.findall("sensor")]}


def quaternion_rotation(quaternion):
    if quaternion is None:
        return np.eye(3)
    q = Gf.Quatd(float(quaternion.GetReal()), Gf.Vec3d(*quaternion.GetImaginary()))
    rot = Gf.Rotation(q)
    return np.column_stack([np.array(rot.TransformDir(Gf.Vec3d(*basis))) for basis in np.eye(3)])


def all_prims(stage):
    return list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))


def load_usd(path, scene_meters_per_unit):
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError(f"Cannot open USD: {path}")
    cache, bodies, joints, colliders, frames, filtered = UsdGeom.XformCache(), {}, {}, [], [], []
    authored_mpu = UsdGeom.GetStageMetersPerUnit(stage)
    kgpu = UsdPhysics.GetStageKilogramsPerUnit(stage)
    total, weighted = 0.0, np.zeros(3)
    for prim in all_prims(stage):
        name, prim_path = prim.GetName(), str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.MassAPI):
            if name in bodies:
                raise ValueError(f"Duplicate USD body names: {name}")
            api = UsdPhysics.MassAPI(prim)
            mass = api.GetMassAttr().Get()
            center = api.GetCenterOfMassAttr().Get()
            diagonal = api.GetDiagonalInertiaAttr().Get()
            axes = api.GetPrincipalAxesAttr().Get()
            if mass is None or center is None or diagonal is None:
                raise ValueError(f"Missing explicit mass properties: {prim_path}")
            com_known = bool(np.isfinite(np.array(center, dtype=float)).all())
            # USD's (-inf,-inf,-inf) fallback means physics must compute CoM.
            # Use the local origin only for a clearly labelled conditional aggregate.
            # Do not count this body as a verified CoM match.
            effective_center = np.array(center, dtype=float) if com_known else np.zeros(3)
            transform = cache.GetLocalToWorldTransform(prim)
            world_center = np.array(transform.Transform(Gf.Vec3d(*effective_center))) * scene_meters_per_unit
            # Tensor is expressed in the local body frame. Report world linear scales;
            # this asset's rigid body transforms all have unit scale.
            basis = np.column_stack([np.array(transform.TransformDir(Gf.Vec3d(*v))) for v in np.eye(3)])
            scales = np.linalg.norm(basis, axis=0)
            if not np.allclose(scales, np.ones(3), atol=1e-6):
                raise ValueError(f"Non-unit rigid body scale requires additional tensor handling: {prim_path}: {scales}")
            principal = quaternion_rotation(axes)
            tensor = principal @ np.diag(np.array(diagonal, dtype=float)) @ principal.T * kgpu * scene_meters_per_unit ** 2
            row = {"path": prim_path, "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
                   "mass_kg": mass * kgpu, "com_link_m": effective_center * scene_meters_per_unit,
                   "com_known_explicit_or_finite": com_known,
                   "com_resolution": "finite authored value" if com_known else "unknown USD fallback; aggregate assumes local origin, runtime verification needed",
                   "diagonal_inertia_authored": list(diagonal),
                   "principal_axes_wxyz": [float(axes.GetReal()), *list(axes.GetImaginary())],
                   "inertia_link_kg_m2": tensor, "world_scale": scales,
                   "com_world_authored_pose_m": world_center,
                   "tiny_inertia_max_le_1e6": bool(np.max(np.linalg.eigvalsh(tensor)) <= 1.000001e-6)}
            bodies[name] = row
            total += row["mass_kg"]
            weighted += row["mass_kg"] * world_center
        if prim.IsA(UsdPhysics.Joint):
            api = UsdPhysics.Joint(prim)
            b0, b1 = api.GetBody0Rel().GetTargets(), api.GetBody1Rel().GetTargets()
            properties = {a.GetName(): a.Get() for a in prim.GetAttributes()
                          if a.HasAuthoredValueOpinion() and any(s in a.GetName() for s in
                          ("physics:lowerLimit", "physics:upperLimit", "drive:", "physxMimicJoint:"))}
            joints[name] = {"path": prim_path, "type": prim.GetTypeName(), "body0": list(map(str, b0)),
                            "body1": list(map(str, b1)), "schemas": list(prim.GetAppliedSchemas()),
                            "world_anchor": prim.IsA(UsdPhysics.FixedJoint) and (not b0 or not b1),
                            "properties": properties}
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            colliders.append({"path": prim_path, "type": prim.GetTypeName(),
                              "enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                              "approximation": mesh_api.GetApproximationAttr().Get() if prim.IsA(UsdGeom.Mesh) else None,
                              "radius": prim.GetAttribute("radius").Get(), "height": prim.GetAttribute("height").Get()})
        if any(s in name.lower() for s in ("camera", "lidar", "imu", "optical")):
            frames.append({"path": prim_path, "type": prim.GetTypeName(), "schemas": list(prim.GetAppliedSchemas())})
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            filtered.append({"path": prim_path, "targets": list(map(str, UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()))})
    sensor_prims = [{"path": str(p.GetPath()), "type": p.GetTypeName(), "schemas": list(p.GetAppliedSchemas())}
                    for p in all_prims(stage) if p.IsA(UsdGeom.Camera)
                    or "Lidar" in p.GetTypeName() or any("Lidar" in api for api in p.GetAppliedSchemas())]
    result = {"path": str(path), "sha256": sha256(path), "default_prim": str(stage.GetDefaultPrim().GetPath()),
              "authored_meters_per_unit": authored_mpu, "kilograms_per_unit": kgpu,
              "comparison_scene_meters_per_unit": scene_meters_per_unit,
              "body_count": len(bodies), "total_mass_kg": total,
              "whole_robot_com_authored_pose_m_in_scene": weighted / total,
              "whole_robot_com_if_opened_standalone_m": weighted / total * authored_mpu / scene_meters_per_unit,
              "aggregate_com_assumes_local_origin_for": [n for n, b in bodies.items() if not b["com_known_explicit_or_finite"]],
              "bodies": bodies, "joints": joints, "colliders": colliders, "sensor_like_frames": frames,
              "actual_camera_or_lidar_prims": sensor_prims, "filtered_pairs": filtered}
    return result, stage


def compare(urdf, usd):
    expected = {n: row for n, row in urdf["links"].items() if row["has_inertial"]}
    common = sorted(set(expected) & set(usd["bodies"]))
    rows = []
    for name in common:
        a, b = expected[name], usd["bodies"][name]
        mass_error = abs(a["mass_kg"] - b["mass_kg"])
        com_error = np.max(np.abs(a["com_link_m"] - b["com_link_m"]))
        tensor_error = np.max(np.abs(a["inertia_link_kg_m2"] - b["inertia_link_kg_m2"]))
        world_error = np.max(np.abs(a["com_world_zero_q_m"] - b["com_world_authored_pose_m"]))
        tensor_match = np.allclose(a["inertia_link_kg_m2"], b["inertia_link_kg_m2"],
                                   atol=TOLERANCE["tensor_kg_m2_atol"], rtol=TOLERANCE["tensor_rtol"])
        rows.append({"link": name, "urdf_mass_kg": a["mass_kg"], "usd_mass_kg": b["mass_kg"],
                     "mass_abs_error_kg": mass_error, "com_max_abs_error_m": com_error,
                     "full_tensor_max_abs_error_kg_m2": tensor_error, "world_com_max_abs_error_m": world_error,
                     "mass_match": mass_error <= TOLERANCE["mass_kg_atol"],
                     "com_known_explicit_or_finite": b["com_known_explicit_or_finite"],
                     "com_match": b["com_known_explicit_or_finite"] and com_error <= TOLERANCE["com_m_atol"],
                     "full_tensor_match": bool(tensor_match),
                     "world_com_zero_q_match": b["com_known_explicit_or_finite"] and world_error <= TOLERANCE["world_com_m_atol"]})
    return {"tolerance": TOLERANCE, "urdf_inertial_link_count": len(expected), "usd_mass_body_count": len(usd["bodies"]),
            "compared_link_count": len(common), "missing_in_usd": sorted(set(expected) - set(usd["bodies"])),
            "extra_usd_mass_bodies": sorted(set(usd["bodies"]) - set(expected)),
            "urdf_massless_links": sorted(set(urdf["links"]) - set(expected)),
            "unverified_com_links": [row["link"] for row in rows if not row["com_known_explicit_or_finite"]],
            "match_counts": {key: sum(row[key] for row in rows) for key in
                             ("mass_match", "com_match", "full_tensor_match", "world_com_zero_q_match")},
            "maximum_errors": {key: max((row[key] for row in rows), default=None) for key in
                               ("mass_abs_error_kg", "com_max_abs_error_m", "full_tensor_max_abs_error_kg_m2", "world_com_max_abs_error_m")},
            "whole_robot_mass_abs_error_kg": abs(urdf["total_mass_kg"] - usd["total_mass_kg"]),
            "whole_robot_com_delta_m": usd["whole_robot_com_authored_pose_m_in_scene"] - urdf["whole_robot_com_zero_q_m"],
            "per_link": rows}


def to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Gf.Vec3f, Gf.Vec3d)):
        return list(value)
    raise TypeError(f"Not JSON-serializable: {type(value)}")


def focused_layer(source):
    """Copy physics properties only, preserving source paths; no meshes or references."""
    out = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(out, UsdGeom.GetStageMetersPerUnit(source))
    UsdGeom.SetStageUpAxis(out, UsdGeom.GetStageUpAxis(source))
    for prim in all_prims(source):
        relevant_body = prim.HasAPI(UsdPhysics.MassAPI)
        relevant_joint = prim.IsA(UsdPhysics.Joint) and (prim.IsA(UsdPhysics.FixedJoint)
                          or "wheel" in prim.GetName() or "gripper" in prim.GetName())
        relevant_collision = prim.HasAPI(UsdPhysics.CollisionAPI) and "wheel" in str(prim.GetPath())
        relevant_filter = prim.HasAPI(UsdPhysics.FilteredPairsAPI)
        if not (relevant_body or relevant_joint or relevant_collision or relevant_filter):
            continue
        copy = out.OverridePrim(prim.GetPath())
        copy.SetTypeName(prim.GetTypeName())
        # APIs and physics attributes preserve original observed values in the review extract.
        schemas = prim.GetMetadata("apiSchemas")
        if schemas is not None:
            copy.SetMetadata("apiSchemas", schemas)
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if attribute.HasAuthoredValueOpinion() and (name.startswith(("physics:", "drive:", "physxMimicJoint:"))
                                                        or name in ("radius", "height", "axis")):
                copy.CreateAttribute(name, attribute.GetTypeName(), attribute.IsCustom()).Set(attribute.Get())
        for relation in prim.GetRelationships():
            if relation.GetName().startswith(("physics:", "physxMimicJoint:")):
                copy.CreateRelationship(relation.GetName(), relation.IsCustom()).SetTargets(relation.GetTargets())
    return out.GetRootLayer().ExportToString()


def historical_delta(prior_path, current_path):
    if not prior_path.is_file() or not current_path.is_file():
        return {"available": False}
    old, _ = load_usd(prior_path, 1.0)
    new, _ = load_usd(current_path, 1.0)
    changes = []
    for name in sorted(set(old["bodies"]) & set(new["bodies"])):
        a, b = old["bodies"][name], new["bodies"][name]
        if a["diagonal_inertia_authored"] != b["diagonal_inertia_authored"]:
            changes.append({"link": name, "old_diagonal_inertia_authored": a["diagonal_inertia_authored"],
                            "new_diagonal_inertia_authored": b["diagonal_inertia_authored"]})
    return {"available": True, "older_snapshot": {"path": str(prior_path), "sha256": sha256(prior_path)},
            "newer_snapshot": {"path": str(current_path), "sha256": sha256(current_path)},
            "inertia_changes": changes,
            "scope": "Local snapshots confirm changes; commit date/author and manual-edit intent require Git provenance."}


def urdf_history_com_summary(original):
    rows = []
    for path in sorted((PRIOR / "urdf_history").glob("sg2_*.urdf")):
        parsed = load_urdf(path)
        base = parsed["links"].get("base_link", {})
        rows.append({"path": str(path), "sha256": sha256(path),
                     "total_mass_kg": parsed["total_mass_kg"],
                     "whole_robot_com_zero_q_m": parsed["whole_robot_com_zero_q_m"],
                     "base_com_link_m": base.get("com_link_m"),
                     "original_usd_aggregate_com_delta_m": original["whole_robot_com_authored_pose_m_in_scene"] - parsed["whole_robot_com_zero_q_m"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ROOT / "source/cyclo_lab/data/robots/FFW/FFW_SG2.usd")
    parser.add_argument("--candidate", type=Path, default=PRIOR / "regen/v4/FFW_SG2_v2.usd")
    parser.add_argument("--urdf", type=Path, default=PRIOR / "urdf_history/sg2_50f368f.urdf")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "sg2_usd_investigation")
    parser.add_argument("--artifact-name", default="static_audit", help="Basename for result JSON/table.")
    parser.add_argument("--review-name", default="FFW_SG2.physics.review.usda", help="Basename for focused review layer.")
    parser.add_argument("--candidate-label", default="candidate_v4", help="JSON key describing audited candidate.")
    args = parser.parse_args()
    args.original = args.original.resolve()
    args.candidate = args.candidate.resolve()
    args.urdf = args.urdf.resolve()
    for name in (args.artifact_name, args.review_name):
        if Path(name).name != name or name in (".", ".."):
            parser.error("Artifact and review names must be plain basenames")
    original, original_stage = load_usd(args.original, 1.0)
    candidate, candidate_stage = load_usd(args.candidate, 1.0)
    urdf = load_urdf(args.urdf)
    original_compare, candidate_compare = compare(urdf, original), compare(urdf, candidate)
    report = {
        "method": {
            "simulation": False, "physics_engine": "none; static USD and URDF parsing only",
            "pose": "URDF FK at q=0 versus USD authored body transforms (not configured task pose).",
            "inertia": "Compare full 3x3 link-frame tensor: R_inertial I_urdf R_inertial^T; R_principal diag(I_usd) R_principal^T.",
            "units": "Compare authored assets as referenced into an Isaac Lab scene with metersPerUnit=1, kgPerUnit=1 and unit body scale. USD references do not automatically rescale using source metersPerUnit. Original stage metadata .01 differs from meter-like geometry. Standalone source interpretation is reported separately.",
            "hardware_accuracy": "URDF consistency only; no measured hardware mass, calibration, sensor intrinsics or motor specification verified.",
            "match": "Tolerances account for float32 serialization and importer tensor diagonalization; equality is not byte-exact.",
        },
        "sources": {"urdf": {"path": str(args.urdf), "sha256": sha256(args.urdf)},
                    "original": {"path": str(args.original), "sha256": sha256(args.original)},
                    "candidate": {"path": str(args.candidate), "sha256": sha256(args.candidate)}},
        "urdf": urdf, "original": original, args.candidate_label: candidate,
        "original_vs_current_urdf": original_compare, "candidate_vs_current_urdf": candidate_compare,
        "historical_local_snapshot_delta": historical_delta(PRIOR / "usd_history/6_348c726.usd", PRIOR / "usd_history/7_24a10b7.usd"),
        "urdf_history_aggregate_com": urdf_history_com_summary(original),
        "original_equals_local_24a10b7_snapshot": sha256(args.original) == sha256(PRIOR / "usd_history/7_24a10b7.usd"),
        "corrections_to_prior_claims": [
            "Original wheel collision prims are convexDecomposition Mesh, not Cylinder; left/right drive and steer collisions are disabled. Cylinder sinking evidence belongs to regenerated v2 in Isaac Sim 5.1.",
            "Changing inertia does not by itself change the center of mass. Mass/CoM distribution and rotational inertia are distinct issues.",
            "Original whole-robot CoM at q=0 matches the old ba75b8d URDF to about 2e-8 m. Latest URDF centers base CoM at 0 with a motion-stability comment; this changes modeled mass distribution without proving hardware accuracy.",
            "Original USD already includes wrist camera mount Xforms and D405/ZED visual geometry, but has no actual camera or lidar sensor prims.",
            "v4 retains seven fixed joints between bodies; these do not anchor the root to the world.",
            "v4 contains ten 1e-6-inertia gripper bodies (two bases and eight finger links), inherited from the latest URDF.",
            "v4 lidar_l_link has an un-authored CoM sentinel; static aggregate assumes local origin and needs runtime verification or explicit URDF-origin authoring.",
            "Capsules preserve radial wheel height but add axial hemispheres and change contact geometry; this is a temporary contact workaround, not hardware-faithful wheel geometry.",
        ],
    }
    out = args.out_dir.resolve()
    allowed = (ROOT / "sg2_usd_investigation").resolve()
    if out != allowed and allowed not in out.parents:
        raise ValueError(f"Audit output must stay in {allowed}")
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(parents=True, exist_ok=True)
    result_path = out / "results" / (args.artifact_name + ".json")
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=to_json) + "\n")
    table = ["Static audit; q=0 authored pose; composition scene units = m, kg.",
             "link                                  mass error[kg]   CoM error[m]  tensor error[kg m2]   world CoM error[m]  matches(m,c,I,w)"]
    for row in candidate_compare["per_link"]:
        flags = ",".join(str(int(row[k])) for k in ("mass_match", "com_match", "full_tensor_match", "world_com_zero_q_match"))
        table.append(f"{row['link']:38s} {row['mass_abs_error_kg']:14.6g} {row['com_max_abs_error_m']:14.6g} {row['full_tensor_max_abs_error_kg_m2']:20.6g} {row['world_com_max_abs_error_m']:20.6g}  {flags}")
    table.append("\nTolerance: " + json.dumps(TOLERANCE))
    (out / "results" / (args.artifact_name + ".table.txt")).write_text("\n".join(table) + "\n")
    original_text = focused_layer(original_stage)
    candidate_text = focused_layer(candidate_stage)
    header = ("# REVIEW EXTRACT ONLY: incomplete asset, not for simulation. Official files are unchanged.\n"
              "# Active declarations below are copied original physics properties, without geometry.\n"
              "# Source: " + str(args.original) + "\n"
              "# Original sha256: " + sha256(args.original) + "\n"
              "# Changes are appended below as commented candidate properties. Paths/topology also change.\n"
              "# Original has disabled left/right wheel colliders; its colliders are Mesh, not Cylinder.\n"
              "# Mass/CoM and inertia are distinct quantities. Candidate is URDF-consistent, not hardware-validated.\n")
    review = original_text.replace("#usda 1.0\n", "#usda 1.0\n" + header, 1)
    review += "\n# ===== PROPOSED CANDIDATE PROPERTIES: comments only, never applied to official asset =====\n"
    review += "# Candidate source: " + str(args.candidate) + "\n# Candidate sha256: " + sha256(args.candidate) + "\n"
    review += "# No root world anchor; wheel capsules enabled; velocity drives; 5 filtered body pairs.\n"
    review += "# Capsule axial contact shape differs from real cylindrical tire. Re-test in 6.0.1.\n"
    review += "# Ten gripper body inertias remain URDF placeholders (1e-6); no physical inference.\n"
    review += "# Actual sensor prims are enumerated in the matching JSON audit.\n"
    review += "\n".join("# " + line for line in candidate_text.splitlines()) + "\n"
    (out / "review" / args.review_name).write_text(review)
    print(json.dumps({"json": str(result_path), "candidate_matches": candidate_compare["match_counts"],
                      "candidate_compared": candidate_compare["compared_link_count"],
                      "candidate_missing": candidate_compare["missing_in_usd"],
                      "candidate_extra": candidate_compare["extra_usd_mass_bodies"],
                      "candidate_max_errors": candidate_compare["maximum_errors"],
                      "urdf_total_mass_kg": urdf["total_mass_kg"], "original_total_mass_kg": original["total_mass_kg"],
                      "candidate_total_mass_kg": candidate["total_mass_kg"],
                      "urdf_com_zero_q_m": urdf["whole_robot_com_zero_q_m"],
                      "original_com_authored_pose_m": original["whole_robot_com_authored_pose_m_in_scene"],
                      "candidate_com_authored_pose_m": candidate["whole_robot_com_authored_pose_m_in_scene"]}, default=to_json))


if __name__ == "__main__":
    main()
