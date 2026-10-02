"""Read-only scene-query evidence for effective SG2 finger/base colliders.

Call probe_gripper_colliders(stage, "/World/Robot", robot) after World.step().
This starts no simulation, moves no body, edits no USD and saves no source file.
Positive PhysX ray hits prove registered colliders at the queried pose. A miss
does not by itself prove absence; geometry bounds and query support are reported.
"""

from __future__ import annotations

import re

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def _field(hit, names):
    for name in names:
        try:
            value = hit.get(name) if isinstance(hit, dict) else getattr(hit, name)
        except (AttributeError, KeyError, TypeError):
            continue
        if value is not None:
            return str(value)
    return ""


def _nearest_body(prim):
    current = prim
    while current and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def _actual_body_matrices(articulation):
    if articulation is None:
        return {}, "No articulation supplied; bounding boxes use authored stage transforms"
    try:
        names = articulation.body_names
        transforms = articulation._physics_view.get_link_transforms()[0]
        matrices = {}
        for name, values in zip(names, transforms, strict=True):
            rotation = Gf.Rotation(Gf.Quatd(float(values[6]), Gf.Vec3d(*[float(v) for v in values[3:6]])))
            transform = Gf.Transform()
            transform.SetRotation(rotation)
            transform.SetTranslation(Gf.Vec3d(*[float(v) for v in values[:3]]))
            matrices[name] = transform.GetMatrix()
        return matrices, None
    except Exception as error:
        return {}, f"Could not read actual body poses: {type(error).__name__}: {error}"


def _target_name(path):
    match = re.search(r"(gripper_[lr]_rh_p12_rn_(?:[rl][12]|base))(?:/|$)", path)
    if match:
        return match.group(1)
    for side in ("l", "r"):
        if f"/arm_{side}_link7/collisions/base/" in path:
            return f"gripper_{side}_rh_p12_rn_base"
    return None


def _shape_matches(collision_path, mesh_paths):
    if not collision_path:
        return False
    # Accept the exact Mesh or its collision-geometry parent Xform. An arbitrary
    # ancestor rigid body is insufficient to prove a specific merged base shape.
    for mesh_path in mesh_paths:
        geometry_root = mesh_path.split("/collisions/", 1)[0] + "/collisions/" + mesh_path.split("/collisions/", 1)[1].split("/", 1)[0]
        if collision_path == geometry_root or collision_path.startswith(geometry_root + "/"):
            return True
    return False


def probe_gripper_colliders(stage, root_path="/World/Robot", articulation=None):
    """Return JSON-compatible raycast evidence for eight fingers and two bases.

    Runtime body poses come from the optional Core Articulation's PhysX view.
    For each mesh bounding box, rays cross a 3 x 3 grid from both directions on
    each axis. raycast_all avoids another robot shape occluding the target.
    Both dict-style and current RaycastHit object fields are supported.
    """
    result = {
        "method": "PhysX raycast_all; 3-axis 3x3 bbox grid in both directions",
        "read_only": True,
        "root_path": root_path,
        "query_supported": False,
        "actual_body_pose_used": False,
        "targets": {},
        "limitations": ["A no-hit result is inconclusive without query and pose support; no grasp-force or object-retention validation"],
    }
    body_matrices, pose_warning = _actual_body_matrices(articulation)
    result["actual_body_pose_used"] = bool(body_matrices)
    if pose_warning:
        result["limitations"].append(pose_warning)
    cache = UsdGeom.XformCache()
    expected = [f"gripper_{side}_rh_p12_rn_{part}" for side in ("l", "r") for part in ("r1", "r2", "l1", "l2", "base")]
    records = {name: {"points": [], "mesh_paths": [], "body_paths": set(), "bounds_source": set()} for name in expected}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not path.startswith(root_path + "/") or "/collisions/" not in path or not prim.IsA(UsdGeom.Mesh):
            continue
        name = _target_name(path)
        if name not in records:
            continue
        body = _nearest_body(prim)
        matrix = cache.GetLocalToWorldTransform(prim)
        source = "authored_stage_pose"
        if body:
            records[name]["body_paths"].add(str(body.GetPath()))
            if body.GetName() in body_matrices:
                matrix = matrix * cache.GetLocalToWorldTransform(body).GetInverse() * body_matrices[body.GetName()]
                source = "actual_PhysX_body_pose"
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
        records[name]["points"].extend(tuple(matrix.Transform(Gf.Vec3d(point))) for point in points)
        records[name]["mesh_paths"].append(path)
        records[name]["bounds_source"].add(source)

    # This standard USD parser is independent supporting evidence. It does not
    # replace PhysX runtime query results and can reject Xform-authored colliders.
    try:
        parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, [root_path])
        result["usd_parser_shape_counts"] = {str(kind): len(values[0]) for kind, values in parsed.items() if "Shape" in str(kind)}
        result["usd_parser_mesh_paths"] = [str(path) for kind, values in parsed.items() if "MeshShape" in str(kind) for path in values[0] if "gripper" in str(path) or "/collisions/base/" in str(path)]
    except Exception as error:
        result["usd_parser_error"] = f"{type(error).__name__}: {error}"

    try:
        import carb
        from omni.physx import get_physx_scene_query_interface

        query = get_physx_scene_query_interface()
        if not callable(getattr(query, "raycast_all", None)):
            raise RuntimeError("raycast_all is unavailable")
    except Exception as error:
        result["query_error"] = f"{type(error).__name__}: {error}"
        for name, record in records.items():
            result["targets"][name] = {"status": "unsupported_query", "mesh_paths": record["mesh_paths"]}
        return result

    supported_queries = 0
    for name, record in records.items():
        evidence = {"mesh_paths": record["mesh_paths"], "body_paths": sorted(record["body_paths"]), "bounds_source": sorted(record["bounds_source"]), "rays_attempted": 0, "rays_completed": 0, "matching_collision_paths": [], "matching_rigid_body_paths": [], "other_collision_hit_count": 0, "query_errors": []}
        points = record["points"]
        if not points:
            evidence["status"] = "no_authored_collision_mesh_found"
            result["targets"][name] = evidence
            continue
        lo = [min(point[axis] for point in points) for axis in range(3)]
        hi = [max(point[axis] for point in points) for axis in range(3)]
        evidence["bbox_world"] = [lo, hi]
        matching_collisions, matching_bodies = set(), set()

        def report_hit(hit):
            collision = _field(hit, ("collision", "collisionPath", "collision_path"))
            body = _field(hit, ("rigidBody", "rigid_body", "rigidBodyPath", "rigid_body_path"))
            if _shape_matches(collision, record["mesh_paths"]):
                matching_collisions.add(collision)
            elif body in record["body_paths"]:
                matching_bodies.add(body)
            else:
                evidence["other_collision_hit_count"] += 1
            return True

        for axis in range(3):
            transverse = [i for i in range(3) if i != axis]
            for u in (.15, .5, .85):
                for v in (.15, .5, .85):
                    for sign in (1.0, -1.0):
                        origin = [(lo[i] + hi[i]) / 2 for i in range(3)]
                        for i, fraction in zip(transverse, (u, v), strict=True):
                            origin[i] = lo[i] + fraction * (hi[i] - lo[i])
                        margin = .005
                        origin[axis] = lo[axis] - margin if sign > 0 else hi[axis] + margin
                        direction = [0., 0., 0.]
                        direction[axis] = sign
                        distance = max(hi[axis] - lo[axis] + 2 * margin, .01)
                        evidence["rays_attempted"] += 1
                        try:
                            try:
                                query.raycast_all(carb.Float3(*origin), carb.Float3(*direction), distance, report_hit, True)
                            except TypeError:
                                query.raycast_all(carb.Float3(*origin), carb.Float3(*direction), distance, report_hit)
                            evidence["rays_completed"] += 1
                            supported_queries += 1
                        except Exception as error:
                            message = f"{type(error).__name__}: {error}"
                            if message not in evidence["query_errors"]:
                                evidence["query_errors"].append(message)
        evidence["matching_collision_paths"] = sorted(matching_collisions)
        evidence["matching_rigid_body_paths"] = sorted(matching_bodies)
        if matching_collisions:
            evidence["status"] = "confirmed_collision_path_hit"
        elif matching_bodies and not name.endswith("_base"):
            evidence["status"] = "confirmed_unique_finger_body_hit"
        elif matching_bodies:
            evidence["status"] = "rigid_body_hit_without_base_shape_identification"
        elif not evidence["rays_completed"]:
            evidence["status"] = "unsupported_query"
        else:
            evidence["status"] = "no_target_hit"
        result["targets"][name] = evidence
    result["query_supported"] = supported_queries > 0
    confirmed = ("confirmed_collision_path_hit", "confirmed_unique_finger_body_hit")
    result["confirmed_finger_count"] = sum(result["targets"][name]["status"] in confirmed for name in expected if not name.endswith("_base"))
    result["confirmed_base_count"] = sum(result["targets"][name]["status"] == "confirmed_collision_path_hit" for name in expected if name.endswith("_base"))
    result["all_eight_finger_colliders_confirmed"] = result["confirmed_finger_count"] == 8
    result["both_gripper_base_colliders_confirmed"] = result["confirmed_base_count"] == 2
    return result
