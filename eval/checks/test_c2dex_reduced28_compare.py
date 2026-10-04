import json

import numpy as np
from scipy.spatial.transform import Rotation

from analysis.metrics import analyze


def write_rollout(tmp_path, force):
    rotation=Rotation.from_rotvec(np.deg2rad([0,30,60])[:,None]*np.array([0.,0.,1.]))
    quat=rotation.as_quat()[:,[3,0,1,2]]
    pos=np.tile([1.,2.,3.],(3,1))
    canonical=np.tile([.01,0,0],(3,5,1))
    points=np.einsum('tij,tfj->tfi',rotation.as_matrix(),canonical)+pos[:,None,:]
    np.savez(tmp_path/'assembly_trace.npz',object_pos=pos[:,None],object_quat=quat[:,None],
        reference_object_pos=pos,reference_object_quat=quat,contact_pos_w=points[:,None],
        fingertip_force_n=np.full((3,1,5),force),fps=30.)
    (tmp_path/'summary.json').write_text(json.dumps(dict(trials=[dict(physics_valid=True,
        silent_reset_count=0,success=False)],evaluation_contract=dict(policy_checkpoint=None))))
    targets=dict(frame=np.arange(3),finger=np.zeros(3,int),target_object=canonical[:,0])
    return targets


def test_actual_object_frame_contact_transform(tmp_path):
    targets=write_rollout(tmp_path,1.)
    result,_=analyze(tmp_path,targets,np.array([0.,0.,1.]))
    assert result['stable_contact']['measured_samples']==3
    assert result['stable_contact']['centroid_distance_mean_mm']<1e-10
    assert abs(result['axial_net_deg']-60)<1e-10


def test_no_contact_is_missing_not_zero_distance(tmp_path):
    targets=write_rollout(tmp_path,0.)
    result,_=analyze(tmp_path,targets,np.array([0.,0.,1.]))
    assert result['stable_contact']['measured_coverage']==0.
    assert result['stable_contact']['centroid_distance_mean_mm'] is None
    assert result['contact_frames']==0
