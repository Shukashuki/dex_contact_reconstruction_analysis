"""Recompute both ZIP-derived wrist trajectories on the requested 28-DOF robot."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from screwdriver_assembly_contract import REFERENCE, SIDECAR, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-name',default='reduced28')
    a=p.parse_args();root=a.root;repo=root/'baseline_repo'
    urdf=repo/'source/regrind/regrind/assets/tron2_axis180/assembly_bilateral_axis180_reduced28_physicsfix.urdf'
    expected='f8987c6c43e0f0a6fee02d13864b1bd59f0a976383ea8dce2da0dc46ba6c67ae'
    if sha256(urdf)!=expected:raise ValueError('Requested robot identity differs')
    old=dict(np.load(repo/'data/precomputedik/arm_ik_physicsfix_bound.npz'))
    tree=ET.parse(urdf)
    for link in tree.findall('link'):
        for tag in ('visual','collision'):
            for child in link.findall(tag):link.remove(child)
    extension=tree.getroot().find('mujoco')
    if extension is None:extension=ET.SubElement(tree.getroot(),'mujoco')
    compiler=extension.find('compiler')
    if compiler is None:compiler=ET.SubElement(extension,'compiler')
    compiler.set('balanceinertia','true');compiler.set('fusestatic','false')
    model=mujoco.MjSpec.from_string(ET.tostring(tree.getroot(),encoding='unicode')).compile()
    data=mujoco.MjData(model);names=[str(x) for x in old['joint_names']]
    joints=np.asarray([model.joint(n).id for n in names]);addresses=model.jnt_qposadr[joints]
    endpoint=model.body('right_hand_base_link').id
    base_p=old['base_pos']+np.asarray([0,0,-1.20035])
    base_R=Rotation.from_quat(old['base_quat_wxyz'][[1,2,3,0]]).as_matrix()
    def fk(q):
        data.qpos[addresses]=q;mujoco.mj_forward(model,data)
        return base_R@data.xpos[endpoint]+base_p,base_R@data.xmat[endpoint].reshape(3,3)
    verified=[]
    for t in (0,90,180,269):
        xyz,rot=fk(old['arm_joint_pos'][t])
        verified.append(float(np.linalg.norm(xyz-old['target_wrist_pos'][t])))
    if max(verified)>1e-5:raise ValueError(f'Packaged FK differs: {verified}')
    output=root/a.output_name;output.mkdir(exist_ok=False)
    reports={}
    for variant in ('baseline','candidate'):
        destination=output/variant;destination.mkdir()
        shutil.copy2(root/variant/REFERENCE,destination/REFERENCE)
        shutil.copy2(root/variant/'scene_from_object.json',destination/'scene_from_object.json')
        with h5py.File(destination/REFERENCE) as h:
            count=len(h['state/robot_pos']);source_times=np.linspace(0,1,count)
            times=np.linspace(0,1,270)
            positions=Akima1DInterpolator(source_times,h['state/robot_pos'][:],method='makima')(times)
            quats=Slerp(source_times,Rotation.from_quat(h['state/robot_quat'][:][:,[1,2,3,0]]))(times).as_quat()[:,[3,0,1,2]]
        solved=[];fps=[];fqs=[];eps=[];ers=[];costs=[];nfev=[]
        previous=old['arm_joint_pos'][0]
        for pos,quat in zip(positions,quats):
            target_R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            def residual(q):
                actual_p,actual_R=fk(q)
                return np.r_[20*(actual_p-pos),Rotation.from_matrix(target_R.T@actual_R).as_rotvec()]
            result=least_squares(residual,np.clip(previous,*model.jnt_range[joints].T),
                bounds=model.jnt_range[joints].T,max_nfev=150,xtol=1e-9,ftol=1e-9,gtol=1e-9)
            previous=result.x;xyz,rot=fk(previous)
            solved.append(previous.copy());fps.append(xyz.copy());fqs.append(Rotation.from_matrix(rot).as_quat()[[3,0,1,2]])
            eps.append(np.linalg.norm(xyz-pos));ers.append(Rotation.from_matrix(target_R.T@rot).magnitude())
            costs.append(result.cost);nfev.append(result.nfev)
        q=np.asarray(solved);ep=np.asarray(eps);er=np.asarray(ers)
        reference_sha=sha256(destination/REFERENCE)
        payload=dict(old);payload.update(arm_joint_pos=q,arm_joint_vel=np.gradient(q,axis=0)*30,
            frame_indices=np.arange(270),reference_frame_indices=np.linspace(0,count-1,270).round().astype(int),
            timestamps_s=np.arange(270)/30,target_wrist_pos=positions,target_wrist_quat_wxyz=quats,
            fk_wrist_pos=np.asarray(fps),fk_wrist_quat_wxyz=np.asarray(fqs),position_error_m=ep,
            orientation_error_rad=er,solver_success=(ep<=.005)&(er<=.05),solver_cost=np.asarray(costs),
            solver_nfev=np.asarray(nfev),reference_sha256=np.asarray(reference_sha),urdf_sha256=np.asarray(expected))
        report=dict(variant=variant,frames=270,reachable_frames=int(((ep<=.005)&(er<=.05)).sum()),
            position_max_mm=float(ep.max()*1000),orientation_max_deg=float(np.degrees(er.max())),
            packaged_FK_position_max_m=max(verified),base_pos=old['base_pos'].tolist(),
            spawn_position=base_p.tolist(),robot_sha256=expected,reference_sha256=reference_sha,
            interpolation='270-frame makima translation and Slerp, matching MotionLoader time endpoints',
            note='IK reachability diagnostic, not a physics-success or admission gate')
        payload['metrics_json']=np.asarray(json.dumps(report));np.savez_compressed(destination/SIDECAR,**payload)
        report['ik_sha256']=sha256(destination/SIDECAR)
        (destination/'manifest.json').write_text(json.dumps(report,indent=2));reports[variant]=report
    (output/'report.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
