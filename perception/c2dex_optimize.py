"""C2Dex contact/SDF/regularized full-trajectory MANO optimisation.

The public paper specifies loss weights, learning rate and step count, but not
the exact regularizer.  This reproduction anchors pose and wrist motion to the
initializer and penalizes changes in first differences; all choices are stored
in the output metadata and report.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perception.c2dex_reconstruction import (finger_flags, pose_matrix, quat_to_matrix)  # noqa: E402
from perception.contact_sdf import finger_vertex_groups  # noqa: E402
from perception.mano_torch import ManoTorch, axis_angle_to_matrix, load_mano_npz  # noqa: E402


@dataclass(frozen=True)
class OptimConfig:
    learning_rate: float = 0.005
    steps: int = 1000
    lambda_contact: float = 8.0
    lambda_sdf: float = 0.01
    lambda_reg: float = 10.0
    sdf_resolution: int = 48
    sdf_padding_m: float = 0.03
    temporal_fraction: float = 0.1


def sdf_grid(mesh_path: Path, resolution: int, padding_m: float):
    import trimesh

    from scipy.spatial import cKDTree

    cache = (PROJECT_ROOT / "perception/hand_artifacts"
             / f"c2dex_sdf_{mesh_path.stem}_{resolution}_{padding_m:.3f}.npz")
    if cache.exists():
        saved = np.load(cache)
        return saved["grid"], saved["lo"], saved["hi"]
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    scale = .01 if mesh_path.name.endswith("_cm.obj") else 1.0
    mesh.apply_scale(scale)
    lo, hi = mesh.bounds - padding_m * np.array([[1], [-1]])
    xs = np.linspace(lo[0], hi[0], resolution)
    ys = np.linspace(lo[1], hi[1], resolution)
    zs = np.linspace(lo[2], hi[2], resolution)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    query = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    # Vertex-distance SDF is a controlled grid approximation: sign is exact
    # for the watertight experiment mesh, magnitude error is bounded by mesh
    # vertex spacing and grid interpolation. It is >10x faster than trimesh
    # exact triangle proximity and remains differentiable after grid sampling.
    values = cKDTree(mesh.vertices).query(query, k=1)[0].astype(np.float32)
    inside = mesh.contains(query)
    values[inside] *= -1.0       # paper convention: positive outside
    grid = values.reshape(resolution, resolution, resolution)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, grid=grid, lo=lo, hi=hi,
                        method=np.asarray("vertex_distance+watertight_contains"))
    return grid, lo, hi


def sample_sdf(grid: torch.Tensor, points: torch.Tensor,
               lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    normalized = 2.0 * (points - lo) / (hi - lo) - 1.0
    # grid_sample 5D grid order is x,y,z; shape N,Dout,Hout,Wout,3.
    coords = normalized.reshape(1, points.shape[0], points.shape[1], 1, 3)
    return F.grid_sample(grid, coords, mode="bilinear", padding_mode="border",
                         align_corners=True).reshape(points.shape[:2])


def _rotation_matrices(pose7: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor(np.stack([quat_to_matrix(p[3:]) for p in pose7]),
                           dtype=torch.float32, device=device)


def stable_targets(contacts, obj_pose: np.ndarray, device):
    segment = np.asarray(contacts["segment_id"], dtype=int)
    stable_segment = np.asarray(contacts["stable_segment"], dtype=int)
    stable_vertex = np.asarray(contacts["stable_vertex"], dtype=int)
    stable_point = np.asarray(contacts["stable_point_object"], dtype=float)
    frames, vertices, targets = [], [], []
    for t, seg in enumerate(segment):
        select = np.where(stable_segment == seg)[0]
        if not len(select):
            continue
        T_ow = pose_matrix(obj_pose[t])
        world = stable_point[select] @ T_ow[:3, :3].T + T_ow[:3, 3]
        frames.extend([t] * len(select))
        vertices.extend(stable_vertex[select].tolist())
        targets.extend(world.tolist())
    return (torch.as_tensor(frames, dtype=torch.long, device=device),
            torch.as_tensor(vertices, dtype=torch.long, device=device),
            torch.as_tensor(targets, dtype=torch.float32, device=device))


def optimize(input_path: Path, contacts_path: Path, mesh_path: Path,
             output_path: Path, config: OptimConfig, device_name: str = "cuda") -> dict:
    data = np.load(input_path, allow_pickle=True)
    contacts = np.load(contacts_path, allow_pickle=True)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    mano_data = load_mano_npz(str(data["side"]))
    model = ManoTorch(mano_data, device)
    T = len(data["mano_pose"])
    pose0 = torch.as_tensor(data["mano_pose"], dtype=torch.float32, device=device)
    trans0 = torch.as_tensor(data["wrist_pose"][:, :3], dtype=torch.float32, device=device)
    R0 = _rotation_matrices(data["wrist_pose"], device)
    obj_R = _rotation_matrices(data["obj_pose"], device)
    obj_t = torch.as_tensor(data["obj_pose"][:, :3], dtype=torch.float32, device=device)
    betas = torch.as_tensor(data["mano_betas"], dtype=torch.float32, device=device)
    frame_i, vertex_i, contact_target = stable_targets(contacts, data["obj_pose"], device)
    if not len(frame_i):
        raise ValueError("no stable contacts to optimise")

    sdf_np, lo_np, hi_np = sdf_grid(mesh_path, config.sdf_resolution,
                                    config.sdf_padding_m)
    sdf = torch.as_tensor(sdf_np, device=device).reshape(1, 1, *sdf_np.shape)
    lo = torch.as_tensor(lo_np, dtype=torch.float32, device=device)
    hi = torch.as_tensor(hi_np, dtype=torch.float32, device=device)

    pose = pose0.clone().requires_grad_(True)
    delta_rot = torch.zeros((T, 3), device=device, requires_grad=True)
    trans = trans0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pose, delta_rot, trans], lr=config.learning_rate)

    def hand_world():
        full_pose = torch.cat([torch.zeros((T, 3), device=device), pose], dim=1)
        joints, vertices = model(full_pose, betas, torch.zeros_like(trans))
        local = vertices - joints[:, :1]
        R = axis_angle_to_matrix(delta_rot) @ R0
        return torch.einsum("tij,tvj->tvi", R, local) + trans[:, None]

    def losses(vertices):
        selected = vertices[frame_i, vertex_i]
        contact = ((selected - contact_target) ** 2).sum(1).mean()
        canonical = torch.einsum("tji,tvj->tvi", obj_R,
                                 vertices - obj_t[:, None])
        distance = sample_sdf(sdf, canonical, lo, hi)
        penetration = torch.relu(-distance).mean()
        anchor = ((pose - pose0) ** 2).mean() + ((trans - trans0) ** 2).mean()
        anchor = anchor + (delta_rot ** 2).mean()
        if T > 1:
            anchor = anchor + config.temporal_fraction * (
                ((pose[1:] - pose[:-1]) - (pose0[1:] - pose0[:-1])) ** 2).mean()
            anchor = anchor + config.temporal_fraction * (
                ((trans[1:] - trans[:-1]) - (trans0[1:] - trans0[:-1])) ** 2).mean()
            anchor = anchor + config.temporal_fraction * (
                (delta_rot[1:] - delta_rot[:-1]) ** 2).mean()
        total = (config.lambda_contact * contact + config.lambda_sdf * penetration
                 + config.lambda_reg * anchor)
        return total, contact, penetration, anchor

    with torch.no_grad():
        initial = losses(hand_world())
    history = []
    for step in range(config.steps):
        optimizer.zero_grad()
        current = losses(hand_world())
        current[0].backward()
        optimizer.step()
        with torch.no_grad():
            pose.clamp_(-1.8, 1.8)
        if step == 0 or (step + 1) % 100 == 0:
            history.append({"step": step + 1, "total": float(current[0]),
                            "contact": float(current[1]), "sdf": float(current[2]),
                            "reg": float(current[3])})
    with torch.no_grad():
        final = losses(hand_world())
        R_final = (axis_angle_to_matrix(delta_rot) @ R0).cpu().numpy()
    quat_xyzw = Rotation.from_matrix(R_final).as_quat()
    wrist = np.column_stack([trans.detach().cpu().numpy(), quat_xyzw[:, 3],
                             quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]])
    try:
        meta = ast.literal_eval(str(data["meta"]))
    except (ValueError, SyntaxError):
        meta = {}
    meta["c2dex_reconstruction"] = {
        "paper": "arXiv:2608.07045", "implementation": "independent_reproduction",
        "contacts": str(contacts_path), "config": asdict(config),
        "regularizer": "initial-pose/wrist anchor + first-difference change penalty",
    }
    payload = {key: data[key] for key in data.files}
    payload["mano_pose"] = pose.detach().cpu().numpy().astype(np.float32)
    payload["wrist_pose"] = wrist.astype(np.float32)
    payload["contact"] = finger_flags(
        T, np.asarray(contacts["segment_id"], dtype=np.int32),
        np.asarray(contacts["stable_segment"], dtype=np.int32),
        np.asarray(contacts["stable_vertex"], dtype=np.int32),
        finger_vertex_groups(mano_data.weights))
    payload["meta"] = np.asarray(str(meta))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    def metrics(values):
        return {"total": float(values[0]),
                "contact_rms_mm": float(torch.sqrt(values[1]) * 1000),
                "mean_penetration_mm": float(values[2] * 1000),
                "regularizer": float(values[3])}
    report = {"status": "optimization_complete", "device": str(device),
              "input": str(input_path), "contacts": str(contacts_path),
              "output": str(output_path), "config": asdict(config),
              "stable_constraints": int(len(frame_i)),
              "initial": metrics(initial), "final": metrics(final),
              "history": history,
              "reproduction_caveat": "exact C2Dex regularizer is not public"}
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sdf-resolution", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config = OptimConfig(steps=args.steps, sdf_resolution=args.sdf_resolution)
    report = optimize(args.input, args.contacts, args.mesh, args.output, config, args.device)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
