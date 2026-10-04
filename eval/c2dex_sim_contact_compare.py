"""Compare C2Dex object-side contact predictions with MuJoCo contacts.

Both point sets must already be in the same canonical object frame.  C2Dex
predictions are expanded from (segment, MANO vertex) medoids to every frame in
the segment; simulator contacts are MuJoCo narrow-phase ``contact.pos`` points
transformed by the simulated object's pose at that exact substep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRED = PROJECT_ROOT / "perception/output/c2dex_clip_002_left.contacts.npz"
DEFAULT_SIM = PROJECT_ROOT / "retarget/output/debug/rollout_c2dex_contact_replay_left.npz"
DEFAULT_OUT = PROJECT_ROOT / "eval/output/c2dex_sim_contact_compare.json"


def unique_points(points: np.ndarray, radius_m: float = 0.001) -> np.ndarray:
    """Deterministically suppress repeated substep/convex-part contacts."""
    kept: list[np.ndarray] = []
    for point in np.asarray(points, dtype=float):
        if not kept or np.linalg.norm(np.asarray(kept) - point, axis=1).min() > radius_m:
            kept.append(point)
    return np.asarray(kept, dtype=float).reshape(-1, 3)


def prediction_by_frame(data) -> list[np.ndarray]:
    segment = np.asarray(data["segment_id"], dtype=int)
    stable_segment = np.asarray(data["stable_segment"], dtype=int)
    stable_point = np.asarray(data["stable_point_object"], dtype=float)
    by_segment = {s: stable_point[stable_segment == s] for s in np.unique(segment)}
    return [by_segment.get(int(s), np.empty((0, 3))) for s in segment]


def simulation_by_frame(data, frames: int, dedup_m: float) -> list[np.ndarray]:
    contact_frame = np.asarray(data["contact_frame"], dtype=int)
    points = np.asarray(data["contact_point_object"], dtype=float)
    return [unique_points(points[contact_frame == t], dedup_m) for t in range(frames)]


def _summary(values_m: list[float]) -> dict:
    if not values_m:
        return {"count": 0, "mean_mm": None, "median_mm": None, "p90_mm": None}
    v = np.asarray(values_m) * 1000.0
    return {"count": len(v), "mean_mm": float(v.mean()),
            "median_mm": float(np.median(v)), "p90_mm": float(np.percentile(v, 90))}


def compare(predicted: list[np.ndarray], simulated: list[np.ndarray],
            thresholds_mm=(5.0, 10.0, 20.0)) -> dict:
    T = min(len(predicted), len(simulated))
    pred_frames = {t for t in range(T) if len(predicted[t])}
    sim_frames = {t for t in range(T) if len(simulated[t])}
    overlap = sorted(pred_frames & sim_frames)
    pred_to_sim: list[float] = []
    sim_to_pred: list[float] = []
    for t in overlap:
        pred_to_sim.extend(cKDTree(simulated[t]).query(predicted[t], k=1)[0].tolist())
        sim_to_pred.extend(cKDTree(predicted[t]).query(simulated[t], k=1)[0].tolist())
    thresholds = {}
    for mm in thresholds_mm:
        threshold = mm / 1000.0
        thresholds[f"{mm:g}mm"] = {
            "prediction_coverage": (float(np.mean(np.asarray(pred_to_sim) <= threshold))
                                    if pred_to_sim else None),
            "sim_precision": (float(np.mean(np.asarray(sim_to_pred) <= threshold))
                              if sim_to_pred else None),
        }
    p2s, s2p = _summary(pred_to_sim), _summary(sim_to_pred)
    means = [v for v in (p2s["mean_mm"], s2p["mean_mm"]) if v is not None]
    union = pred_frames | sim_frames
    return {
        "frames": T,
        "predicted_contact_frames": len(pred_frames),
        "sim_contact_frames": len(sim_frames),
        "overlap_frames": len(overlap),
        "temporal_iou": len(overlap) / len(union) if union else None,
        "predicted_frame_recall": len(overlap) / len(pred_frames) if pred_frames else None,
        "sim_frame_precision": len(overlap) / len(sim_frames) if sim_frames else None,
        "prediction_to_sim": p2s,
        "sim_to_prediction": s2p,
        "symmetric_chamfer_mean_mm": float(np.mean(means)) if means else None,
        "thresholds": thresholds,
        "scope": "object-canonical point-set agreement; cross-embodiment, no finger correspondence",
    }


def run(prediction_path: Path, simulation_path: Path) -> dict:
    prediction = np.load(prediction_path, allow_pickle=True)
    simulation = np.load(simulation_path, allow_pickle=True)
    required = {"contact_frame", "contact_point_object"}
    missing = sorted(required - set(simulation.files))
    if missing:
        raise ValueError(f"simulation artifact lacks 3D contacts: {missing}; rerun physics replay")
    pred = prediction_by_frame(prediction)
    sim = simulation_by_frame(simulation, len(pred), dedup_m=0.001)
    report = compare(pred, sim)
    if "obj_sim" in simulation.files:
        obj_sim = np.asarray(simulation["obj_sim"], dtype=float)
        held = [t for t in range(min(len(pred), len(obj_sim)))
                if obj_sim[t, 2] - obj_sim[0, 2] > 0.03 and len(sim[t])]
        held_set = set(held)
        empty = np.empty((0, 3))
        held_report = compare(
            [points if t in held_set else empty for t, points in enumerate(pred)],
            [points if t in held_set else empty for t, points in enumerate(sim)])
        held_report["frame_indices"] = held
        held_report["criterion"] = "sim object >3cm above initial and hand-object contact present"
        report["held_aloft_comparison"] = held_report
    report.update({"prediction": str(prediction_path), "simulation": str(simulation_path),
                   "sim_raw_contact_points": int(len(simulation["contact_frame"])),
                   "sim_unique_contact_points": int(sum(map(len, sim))),
                   "dedup_radius_mm": 1.0})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--simulation", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    report = run(args.prediction, args.simulation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
