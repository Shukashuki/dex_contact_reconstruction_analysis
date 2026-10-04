"""Paper-faithful C2Dex HOI reconstruction for an existing HandObjectTraj.

This is an independent reproduction of arXiv:2608.07045, not the authors'
implementation (their public repository does not yet contain code).  It keeps
the paper's observable stages explicit:

1. silhouette-ray object-side contact observations + normal filtering;
2. locally stable hand/object segments;
3. per-MANO-vertex DBSCAN and dominant-cluster medoids in object space;
4. export of stable contacts and schema-compatible per-finger contact flags.

The unreleased thresholds are named CLI/config parameters and are recorded in
the output, rather than being presented as official C2Dex values.  The current
entry point deliberately stops before trajectory optimisation; ``status`` in
the report makes that boundary machine-readable.

Example (the TACO sequence used as dexpipe's golden clip)::

  python -m perception.c2dex_reconstruction \
    --input perception/output/golden_clip_002_left.npz \
    --clip data/golden/clip_002 --stride 5
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perception.contact_sdf import FINGER_ORDER, finger_vertex_groups  # noqa: E402
from perception.mano_layer import forward, load_mano  # noqa: E402


@dataclass(frozen=True)
class C2DexConfig:
    # The paper defines these tests but does not publish their values.
    normal_threshold: float = 0.5
    articulation_threshold_rad: float = 0.35
    relative_translation_threshold_m: float = 0.015
    relative_rotation_threshold_rad: float = 0.20
    dbscan_eps_m: float = 0.008
    dbscan_min_samples: int = 3
    # The paper uses silhouette overlap without a ray-depth gap cutoff.
    # A finite field remains for ablations; 10 m is effectively disabled.
    max_ray_gap_m: float = 10.0


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q), 1e-12, None)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = quat_to_matrix(pose[3:])
    out[:3, 3] = pose[:3]
    return out


def rotation_angle(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1, 1)))


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, length, out=np.zeros_like(normals), where=length > 1e-12)


def hand_meshes(data, model) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild schema MANO parameters into world vertices and normals."""
    T = len(data["mano_pose"])
    pose = np.concatenate([np.zeros((T, 3)), data["mano_pose"]], axis=1)
    _, local = forward(model, pose, data["mano_betas"], center_idx=0)
    local_normals = np.stack([vertex_normals(v, model.faces) for v in local])
    vertices = np.empty_like(local)
    normals = np.empty_like(local_normals)
    for t in range(T):
        R = quat_to_matrix(data["wrist_pose"][t, 3:])
        vertices[t] = local[t] @ R.T + data["wrist_pose"][t, :3]
        normals[t] = local_normals[t] @ R.T
    return vertices, normals


def stable_segments(mano_pose: np.ndarray, wrist_pose: np.ndarray,
                    obj_pose: np.ndarray, config: C2DexConfig) -> np.ndarray:
    """Maximal contiguous locally stable segments (paper Sec. III-A)."""
    T = len(mano_pose)
    segment = np.zeros(T, dtype=np.int32)
    relative = [np.linalg.inv(pose_matrix(obj_pose[t])) @ pose_matrix(wrist_pose[t])
                for t in range(T)]
    for t in range(1, T):
        dtheta = np.linalg.norm(mano_pose[t] - mano_pose[t - 1])
        delta = np.linalg.inv(relative[t - 1]) @ relative[t]
        stable = (dtheta < config.articulation_threshold_rad
                  and np.linalg.norm(delta[:3, 3]) < config.relative_translation_threshold_m
                  and rotation_angle(delta[:3, :3]) < config.relative_rotation_threshold_rad)
        segment[t] = segment[t - 1] + (not stable)
    return segment


def dbscan_labels(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Small deterministic DBSCAN implementation; -1 denotes noise."""
    points = np.asarray(points, dtype=np.float64)
    neighbours = cKDTree(points).query_ball_point(points, eps)
    labels = np.full(len(points), -1, dtype=np.int32)
    visited = np.zeros(len(points), dtype=bool)
    cluster = 0
    for seed in range(len(points)):
        if visited[seed]:
            continue
        visited[seed] = True
        if len(neighbours[seed]) < min_samples:
            continue
        labels[seed] = cluster
        queue = list(neighbours[seed])
        queued = set(queue)
        while queue:
            item = queue.pop(0)
            if not visited[item]:
                visited[item] = True
                if len(neighbours[item]) >= min_samples:
                    for nxt in neighbours[item]:
                        if nxt not in queued:
                            queue.append(nxt)
                            queued.add(nxt)
            if labels[item] < 0:
                labels[item] = cluster
        cluster += 1
    return labels


def dominant_medoid(points: np.ndarray, eps: float,
                    min_samples: int) -> np.ndarray | None:
    labels = dbscan_labels(points, eps, min_samples)
    valid = labels[labels >= 0]
    if not len(valid):
        return None
    ids, counts = np.unique(valid, return_counts=True)
    # np.argmax makes equal-size ties deterministic (lowest cluster id).
    cluster = ids[np.argmax(counts)]
    members = points[labels == cluster]
    distance = np.linalg.norm(members[:, None] - members[None], axis=2)
    return members[np.argmin(distance.sum(axis=1))]


def stabilize_contacts(observations: list[tuple[int, int, np.ndarray]],
                       segments: np.ndarray, config: C2DexConfig
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return segment ids, MANO vertex ids, canonical medoids."""
    buckets: dict[tuple[int, int], list[np.ndarray]] = {}
    for frame, vertex, point in observations:
        buckets.setdefault((int(segments[frame]), int(vertex)), []).append(point)
    keys, medoids = [], []
    for key in sorted(buckets):
        medoid = dominant_medoid(np.asarray(buckets[key]), config.dbscan_eps_m,
                                 config.dbscan_min_samples)
        if medoid is not None:
            keys.append(key)
            medoids.append(medoid)
    if not keys:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty((0, 3), np.float32))
    return (np.asarray([k[0] for k in keys], np.int32),
            np.asarray([k[1] for k in keys], np.int32),
            np.asarray(medoids, np.float32))


def ray_contact_observations(hand_vertices: np.ndarray,
                             hand_normals: np.ndarray,
                             object_vertices_m: np.ndarray,
                             object_faces: np.ndarray,
                             obj_pose: np.ndarray,
                             camera_world_to_cam: np.ndarray,
                             frames: np.ndarray,
                             config: C2DexConfig
                             ) -> list[tuple[int, int, np.ndarray]]:
    """C2Dex silhouette-overlap ray contacts, stored in canonical object space."""
    import trimesh

    mesh = trimesh.Trimesh(object_vertices_m, object_faces, process=False)
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    face_normals = np.asarray(mesh.face_normals)
    observations: list[tuple[int, int, np.ndarray]] = []
    for t in frames:
        T_ow = pose_matrix(obj_pose[t])
        T_wo = np.linalg.inv(T_ow)
        T_cw = camera_world_to_cam[t]
        camera_world = np.linalg.inv(T_cw)[:3, 3]
        origin_o = (T_wo[:3, :3] @ camera_world + T_wo[:3, 3])
        hand_o = hand_vertices[t] @ T_wo[:3, :3].T + T_wo[:3, 3]
        rays = hand_o - origin_o
        ray_length = np.linalg.norm(rays, axis=1)
        rays /= np.clip(ray_length[:, None], 1e-12, None)
        location, ray_id, triangle_id = intersector.intersects_location(
            np.broadcast_to(origin_o, rays.shape), rays, multiple_hits=False)
        for point, vertex, triangle in zip(location, ray_id, triangle_id):
            gap = abs(float(np.linalg.norm(point - origin_o) - ray_length[vertex]))
            object_normal_world = T_ow[:3, :3] @ face_normals[triangle]
            score = -float(hand_normals[t, vertex] @ object_normal_world)
            if score > config.normal_threshold and gap <= config.max_ray_gap_m:
                observations.append((int(t), int(vertex), point.astype(np.float32)))
    return observations


def finger_flags(T: int, segments: np.ndarray, stable_segment: np.ndarray,
                 stable_vertex: np.ndarray, groups: dict[str, np.ndarray]) -> np.ndarray:
    owner = np.full(778, -1, dtype=np.int32)
    for finger, name in enumerate(FINGER_ORDER):
        owner[groups[name]] = finger
    flags = np.zeros((T, 5), dtype=np.float32)
    for seg, vertex in zip(stable_segment, stable_vertex):
        if owner[vertex] >= 0:
            flags[segments == seg, owner[vertex]] = 1.0
    return flags


def _metadata(data) -> dict:
    try:
        return ast.literal_eval(str(data["meta"]))
    except (ValueError, SyntaxError):
        return {}


def _mesh_id(data, meta: dict) -> str:
    if meta.get("obj_id_for_side"):
        return str(meta["obj_id_for_side"])
    obj_id = str(data["obj_id"])
    digits = "".join(ch for ch in obj_id if ch.isdigit())
    if len(digits) >= 3:
        return digits[:3]
    raise ValueError("cannot map obj_id to a clip mesh; pass compatible metadata")


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh
    mesh = trimesh.load(path, process=False, force="mesh")
    scale = 0.01 if path.name.endswith("_cm.obj") else 1.0
    return np.asarray(mesh.vertices, np.float64) * scale, np.asarray(mesh.faces, np.int64)


def run(input_path: Path, clip: Path, output_prefix: Path,
        config: C2DexConfig, stride: int = 1) -> dict:
    data = np.load(input_path, allow_pickle=True)
    meta = _metadata(data)
    mesh_id = _mesh_id(data, meta)
    mesh_path = clip / "objects" / f"{mesh_id}_cm.obj"
    vertices_o, faces_o = load_obj(mesh_path)
    model = load_mano(str(data["side"]))
    hand_v, hand_n = hand_meshes(data, model)
    with h5py.File(clip / "joints.hdf5", "r") as handle:
        extrinsic = handle["camera_extrinsic_world_to_cam"][:]
    T = len(hand_v)
    frames = np.arange(0, T, stride, dtype=np.int32)
    segments = stable_segments(data["mano_pose"], data["wrist_pose"],
                               data["obj_pose"], config)
    observations = ray_contact_observations(
        hand_v, hand_n, vertices_o, faces_o, data["obj_pose"], extrinsic,
        frames, config)
    stable_seg, stable_vertex, stable_point = stabilize_contacts(
        observations, segments, config)
    groups = finger_vertex_groups(model.weights)
    contact = finger_flags(T, segments, stable_seg, stable_vertex, groups)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output_prefix.with_suffix(".contacts.npz")
    np.savez_compressed(sidecar, segment_id=segments,
                        stable_segment=stable_seg, stable_vertex=stable_vertex,
                        stable_point_object=stable_point,
                        observation_frame=np.asarray([x[0] for x in observations], np.int32),
                        observation_vertex=np.asarray([x[1] for x in observations], np.int32),
                        observation_point_object=np.asarray([x[2] for x in observations], np.float32),
                        config=np.asarray(json.dumps(asdict(config), sort_keys=True)))
    report = {
        "status": "contact_stabilization_complete_optimization_pending",
        "paper": "arXiv:2608.07045",
        "input": str(input_path), "clip": str(clip), "mesh": str(mesh_path),
        "frames": T, "sampled_frames": len(frames),
        "segments": int(segments.max()) + 1,
        "raw_observations": len(observations),
        "stable_vertex_segment_pairs": len(stable_vertex),
        "contact_frames": int(contact.any(axis=1).sum()),
        "per_finger_frames": {name: int(contact[:, i].sum())
                              for i, name in enumerate(FINGER_ORDER)},
        "sidecar": str(sidecar), "config": asdict(config),
        "unreleased_by_authors": ["thresholds", "DBSCAN parameters",
                                   "regularizer definition", "experiment split"],
    }
    output_prefix.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path,
                        default=PROJECT_ROOT / "perception/output/c2dex_clip_002_left")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args(argv)
    report = run(args.input, args.clip, args.output_prefix, C2DexConfig(), args.stride)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["stable_vertex_segment_pairs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
