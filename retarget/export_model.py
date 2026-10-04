"""Export FK constants from a locally supplied MuJoCo robot and explicit mapping."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def export(output,xml,contract):
    import mujoco
    mapping=json.loads(Path(contract).read_text())
    model=mujoco.MjModel.from_xml_path(str(xml))
    joints=np.asarray([model.joint(name).id for name in mapping['joint_names']])
    if np.any(model.jnt_type[joints]!=mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError('Export supports scalar hinge joints only')
    addresses=model.jnt_qposadr[joints]
    body_joint=np.full(model.nbody,-1,int)
    for index,joint in enumerate(joints):
        body=int(model.jnt_bodyid[joint])
        if body_joint[body]>=0:raise ValueError('One controlled hinge per body is required')
        body_joint[body]=index
    nodes=np.asarray([model.body(name).id for name in mapping['node_body_names']])
    limits=np.asarray(mapping.get('joint_limits',model.jnt_range[joints]))
    surfaces=[];surface_body=[]
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):continue
        mid=int(model.geom_dataid[geom])
        if model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH or mid<0:continue
        start,count=int(model.mesh_vertadr[mid]),int(model.mesh_vertnum[mid])
        vertices=model.mesh_vert[start:start+count].astype(float)
        matrix=Rotation.from_quat(np.roll(model.geom_quat[geom],-1)).as_matrix()
        surfaces.append(vertices@matrix.T+model.geom_pos[geom])
        surface_body.extend([int(model.geom_bodyid[geom])]*count)
    if not surfaces:raise ValueError('A collision mesh surface is required')
    test_q=np.random.default_rng(7).uniform(*limits.T,size=(3,len(joints)))
    positions=[]
    for q in test_q:
        state=mujoco.MjData(model);state.qpos[addresses]=q;mujoco.mj_forward(model,state)
        positions.append(state.xpos.copy())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,parent=model.body_parentid,pos=model.body_pos,quat=model.body_quat,
        body_joint=body_joint,axis=model.jnt_axis[joints],anchor=model.jnt_pos[joints],limits=limits,
        nodes=nodes,human_ids=mapping['human_ids'],surface=np.concatenate(surfaces),surface_body=surface_body,
        C=mapping['wrist_to_handroot_rotation'],test_q=test_q,test_positions=positions,
        body_names=[model.body(i).name for i in range(model.nbody)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xml',type=Path,required=True);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    export(a.output,a.xml,a.contract)


if __name__=='__main__':main()
