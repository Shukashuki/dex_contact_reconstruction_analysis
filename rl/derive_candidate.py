"""Bind handroot retarget output to a ZIP HDF without changing object/MANO reference."""
import argparse
import json
from pathlib import Path
import shutil
import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from screwdriver_assembly_contract import REFERENCE, SIDECAR


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--robot-model',type=Path,required=True);a=p.parse_args();root=a.root
    from retarget.c2dex_retarget import RobotFK
    with np.load(root/'optimized_robot.npz') as data:
        wrist=data['wrist_pose_W'];joints=data['hand_qpos']
    with np.load(root/'baseline'/SIDECAR,allow_pickle=False) as data:indices=data['reference_frame_indices']
    candidate=root/'candidate';candidate.mkdir(exist_ok=False)
    shutil.copy2(root/'baseline'/REFERENCE,candidate/REFERENCE)
    shutil.copy2(root/'baseline/scene_from_object.json',candidate/'scene_from_object.json')
    with h5py.File(candidate/REFERENCE,'r+') as h:
        count=len(h['state/robot_pos']);time=np.arange(count)
        if len(wrist)!=len(indices):raise ValueError('Retarget frame indices differ')
        hand=np.stack([np.interp(time,indices,joints[:,j]) for j in range(joints.shape[1])],axis=1)
        pos=np.stack([np.interp(time,indices,wrist[:,j]) for j in range(3)],axis=1)
        quat=Slerp(indices,Rotation.from_quat(wrist[:,[4,5,6,3]]))(time).as_quat()[:,[3,0,1,2]]
        with np.load(a.robot_model) as constants:
            fk=RobotFK(constants,'cpu');local=fk(fk.tensor(hand))[1].detach().numpy()
            names=[v.decode() if isinstance(v,bytes) else str(v) for v in h['meta/keypoint_names'][:]]
            body_names=list(constants['body_names']);nodes=[body_names.index(name) for name in names]
        matrices=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
        keypoints=np.einsum('tij,tnj->tni',matrices,local[:,nodes])+pos[:,None,:]
        h['state/robot_joints'][...]=hand;h['state/robot_pos'][...]=pos;h['state/robot_quat'][...]=quat
        for key in ('state/robot_keypoints','reference/robot_keypoints'):h[key][...]=keypoints
    with h5py.File(root/'baseline'/REFERENCE) as b,h5py.File(candidate/REFERENCE) as c:
        for key in ('reference/object_pos','reference/object_quat','state/object_pos','state/object_quat','reference/mano_joint_coords'):
            if not np.array_equal(b[key][:],c[key][:]):raise ValueError('Immutable reference changed: '+key)
    print('Candidate HDF ready; recompute BOTH variants on the 28-DOF robot before physics/PPO.')


if __name__=='__main__':main()
