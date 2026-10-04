"""Compare screwdriver axial rotation, full orientation and contact persistence.

Axial twist uses swing-twist decomposition around the canonical mesh principal
axis, referenced to the same initial demonstration orientation. It is a free
object rotation metric, not a screw-engagement or tightening-torque test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def principal_axis(mesh: Path) -> np.ndarray:
    with mesh.open() as handle:
        vertices = np.asarray([[float(x) for x in line.split()[1:4]]
                               for line in handle if line.startswith('v ')])
    _, _, axes = np.linalg.svd(vertices-vertices.mean(0), full_matrices=False)
    axis = axes[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return axis


def pose_rotations(pose):
    return Rotation.from_quat(np.roll(pose[:, 3:], -1, axis=1))


def twist(rotations, initial, axis):
    q = (initial.inv()*rotations).as_quat()
    projected = q[:, :3] @ axis
    defined = q[:, 3]**2 + projected**2 > 1e-8
    angle = np.unwrap(2*np.arctan2(projected, q[:, 3]))
    angle -= round(angle[0]/(2*np.pi))*(2*np.pi)
    return angle, defined


def compare(rollout: Path, axis, active_speed_deg_s: float = 10.0):
    with np.load(rollout, allow_pickle=True) as data:
        reference = pose_rotations(data['obj_ref'])
        simulated = pose_rotations(data['obj_sim'])
        fps = float(data['fps'])
        ref_angle, ref_defined = twist(reference, reference[0], axis)
        sim_angle, sim_defined = twist(simulated, reference[0], axis)
        angular_error = np.degrees((reference.inv()*simulated).magnitude())
        velocity = np.degrees(np.gradient(ref_angle))*fps
        active = np.abs(velocity) >= active_speed_deg_s
        valid = ref_defined & sim_defined
        selected = active & valid
        difference_deg = np.degrees(sim_angle-ref_angle)
        def metrics(mask):
            if not mask.any():
                return {'frames': 0, 'twist_mae_deg': None, 'twist_rmse_deg': None,
                        'orientation_mean_deg': None, 'contact_frames': 0}
            contact_frames = np.unique(data['contact_frame']) if 'contact_frame' in data.files else []
            return {'frames': int(mask.sum()),
                    'twist_mae_deg': float(np.abs(difference_deg[mask]).mean()),
                    'twist_rmse_deg': float(np.sqrt(np.mean(difference_deg[mask]**2))),
                    'orientation_mean_deg': float(angular_error[mask].mean()),
                    'contact_frames': int(np.isin(np.where(mask)[0], contact_frames).sum())}
        report = {'rollout': str(rollout), 'fps': fps, 'frames': len(reference),
                  'axis_object': np.asarray(axis).tolist(),
                  'reference_axial_net_deg': float(np.degrees(ref_angle[-1]-ref_angle[0])),
                  'reference_axial_travel_deg': float(np.degrees(np.abs(np.diff(ref_angle)).sum())),
                  'sim_axial_net_deg': float(np.degrees(sim_angle[-1]-sim_angle[0])),
                  'sim_axial_travel_deg': float(np.degrees(np.abs(np.diff(sim_angle)).sum())),
                  'initial_orientation_error_deg': float(angular_error[0]),
                  'twist_undefined_frames': int((~valid).sum()),
                  'all_frames': metrics(valid), 'rotation_active_frames': metrics(selected),
                  'active_criterion_deg_s': active_speed_deg_s,
                  'rotation_active_frame_indices': np.where(selected)[0].tolist()}
        trace = {'time_s': np.arange(len(reference))/fps,
                 'ref_twist_deg': np.degrees(ref_angle), 'sim_twist_deg': np.degrees(sim_angle),
                 'orientation_error_deg': angular_error, 'rotation_active': active}
        return report, trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--topo-rollout', type=Path, required=True)
    p.add_argument('--c2dex-rollout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    axis = principal_axis(args.mesh)
    topo, tt = compare(args.topo_rollout, axis)
    c2dex, ct = compare(args.c2dex_rollout, axis)
    with np.load(args.topo_rollout) as a, np.load(args.c2dex_rollout) as b:
        if not np.allclose(a['obj_ref'], b['obj_ref']):
            raise ValueError('reference object trajectory differs')
    result = {'topo': topo, 'c2dex_kinematic': c2dex,
              'scope': 'same-reference free-object axial rotation; not screw tightening',
              'metric': 'swing-twist around positive canonical principal axis, unwrapped, shared reference R0'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.output.with_suffix('.trace.npz'), **tt,
                        c2dex_sim_twist_deg=ct['sim_twist_deg'],
                        c2dex_orientation_error_deg=ct['orientation_error_deg'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
