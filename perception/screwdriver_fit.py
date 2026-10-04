"""Fit ZIP 3D keypoints and stabilize proximity contacts, without camera rays."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from screwdriver_assembly_contract import REFERENCE, SIDECAR


def prepare(archive, mesh, scene, root):
    raw=archive.read_bytes()
    expected='4e2735400b5e44a28ac6f05bda926693ed2944c6dec787b1f7c8b0c32f8f92cb'
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Not the recorded screwdriver ZIP')
    root.mkdir(parents=True,exist_ok=False);baseline=root/'baseline';baseline.mkdir()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in (REFERENCE,SIDECAR):
            candidates=[n for n in z.namelist() if Path(n).name==name]
            if len(candidates)!=1:raise ValueError('Expected one '+name)
            (baseline/name).write_bytes(z.read(candidates[0]))
    shutil.copy2(mesh,root/'screwdriver_object_m.obj')
    shutil.copy2(scene,baseline/'scene_from_object.json')
    with np.load(baseline/SIDECAR,allow_pickle=False) as ik:indices=ik['reference_frame_indices'].astype(int)
    with h5py.File(baseline/REFERENCE) as h:
        values={name:h[name][:][indices] for name in ('reference/mano_joint_coords',
            'reference/object_pos','reference/object_quat','state/robot_pos','state/robot_quat','state/robot_joints')}
        ids=h['meta/keypoint_mano_ids'][:].astype(int)
    np.savez_compressed(root/'reference_arrays.npz',**values,indices=indices)
    np.savez_compressed(root/'baseline_robot.npz',hand_qpos=values['state/robot_joints'],
        wrist_pose_W=np.c_[values['state/robot_pos'],values['state/robot_quat']],
        obj_pose_W=np.c_[values['reference/object_pos'],values['reference/object_quat']],
        fps=30.,embodiment='revo3',source='front_table_zip_reference',schema_version='1.0')
    clip=root/'human_clip';clip.mkdir()
    with h5py.File(clip/'joints.hdf5','w') as h:h['right_hand']=values['reference/mano_joint_coords'][:,ids]


def fit(root: Path):
    import torch
    import trimesh
    from perception.mano_torch import load_mano_npz, fit_sequence_trf
    from perception.c2dex_reconstruction import C2DexConfig, stable_segments, stabilize_contacts, vertex_normals
    torch.set_num_threads(4)
    source = np.load(root / 'reference_arrays.npz')
    raw_joints = source['reference/mano_joint_coords']
    mano = load_mano_npz('right')
    rest = mano.J_regressor @ mano.v_template
    parents = mano.parents.astype(int)[1:]
    bone_model = np.linalg.norm(rest[1:] - rest[parents], axis=-1)
    bone_zip = np.linalg.norm(raw_joints[:, 1:16] - raw_joints[:, parents], axis=-1)
    scale = float(np.sum(bone_zip * bone_model) / np.sum(np.broadcast_to(bone_model, bone_zip.shape)**2))
    origins = raw_joints[:, :1]
    fitted = fit_sequence_trf(mano, (raw_joints-origins)/scale, max_iter=120, verbose=True)
    full_pose = fitted['pose']
    fitted_root = origins[:, 0] + scale*(rest[0] + fitted['trans'])
    fitted_vertices = origins + scale*fitted['vertices']
    fitted['fit_mm'] *= scale
    wrist = np.column_stack([fitted_root, Rotation.from_rotvec(full_pose[:, :3]).as_quat()[:, [3,0,1,2]]])
    obj_pose = np.column_stack([source['reference/object_pos'], source['reference/object_quat']])
    mesh = trimesh.load(root / 'screwdriver_object_m.obj', process=False, force='mesh')
    config = C2DexConfig()
    segments = stable_segments(full_pose[:, 3:], wrist, obj_pose, config)
    observations = []
    for t, vertices in enumerate(fitted_vertices):
        R = Rotation.from_quat(np.roll(obj_pose[t, 3:], -1)).as_matrix()
        canonical = (vertices - obj_pose[t, :3]) @ R
        near, distance, face = trimesh.proximity.closest_point(mesh, canonical)
        normals = vertex_normals(vertices, mano.faces) @ R
        compatible = -(normals * mesh.face_normals[face]).sum(-1) > .5
        for v in np.where((distance <= .008) & compatible)[0]:
            observations.append((t, int(v), near[v]))
    ss, sv, sp = stabilize_contacts(observations, segments, config)
    if not len(sv):
        raise ValueError('No geometric stable contacts; cannot run contact-preserving retarget')
    np.savez_compressed(root / 'stable_contacts.npz', segment_id=segments,
                        stable_segment=ss, stable_vertex=sv, stable_point_object=sp)
    np.savez_compressed(root / 'human.npz', fps=30., side='right', mano_pose=full_pose[:, 3:],
                        mano_betas=np.zeros(10), mano_scale=scale, wrist_pose=wrist, obj_pose=obj_pose)
    report = {'method':'ZIP 3D MANO-keypoint fit and proximity/opposite-normal contacts; camera rays unavailable',
              'fit_mean_mm':float(np.mean(fitted['fit_mm'])), 'fit_median_mm':float(np.median(fitted['fit_mm'])),
              'fit_p90_mm':float(np.percentile(fitted['fit_mm'],90)),
              'proximity_threshold_mm':8., 'normal_score_threshold':.5,
              'raw_contacts':len(observations), 'stable_vertex_segment_pairs':len(sv),
              'segments':int(segments.max()+1), 'MANO_betas':'zero; shape not present in ZIP',
              'MANO_scale':scale, 'scale_method':'least-squares 16-joint bone lengths, ZIP-only; targets unchanged',
              'status':'3D_contact_input_ready_not_monocular_reconstruction'}
    (root / 'contact_input_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--archive',type=Path)
    p.add_argument('--mesh',type=Path);p.add_argument('--scene',type=Path)
    a=p.parse_args()
    if a.prepare:
        if not all((a.archive,a.mesh,a.scene)):p.error('--prepare needs --archive --mesh --scene')
        prepare(a.archive,a.mesh,a.scene,a.root)
    else:fit(a.root)


if __name__=='__main__':main()
