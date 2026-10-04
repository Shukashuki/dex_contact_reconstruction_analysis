"""Task-local ZIP-only adaptation of the user-selected dex-rl baseline.

Keeps its 28-DOF asset, action implementation, original reward family and PPO
architecture. Evaluation is nominal/no-DR; paired fine-tuning is a bounded
nominal/no-DR adaptation, not reproduction of the original two-rank training.
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np

from screwdriver_assembly_contract import REFERENCE, SIDECAR, sha256, workspace_bounds


def configure(cfg, inputs, output, num_envs, seed):
    from regrind.robots.tron2_revo3_assembly import ASSEMBLY_URDF_PATH, ASSEMBLY_URDF_SHA256
    inputs=Path(inputs);output=Path(output)
    info=json.loads((inputs/'manifest.json').read_text())
    if sha256(inputs/REFERENCE)!=info['reference_sha256'] or sha256(inputs/SIDECAR)!=info['ik_sha256']:
        raise ValueError('Input identity differs')
    if sha256(ASSEMBLY_URDF_PATH)!=info['robot_sha256'] or info['robot_sha256']!=ASSEMBLY_URDF_SHA256:
        raise ValueError('Robot identity differs')
    with np.load(inputs/SIDECAR) as ik:
        root=ik['base_pos']+np.asarray([0,0,-1.20035]);quat=ik['base_quat_wxyz']
        arm={str(n):float(q) for n,q in zip(ik['joint_names'],ik['arm_joint_pos'][0])}
    with h5py.File(inputs/REFERENCE) as h:
        names=[x.decode() if isinstance(x,bytes) else str(x) for x in h['meta/joint_names'][:]]
        hand=dict(zip(names,map(float,h['state/robot_joints'][0])))
        lower,upper=workspace_bounds(h['reference/object_pos'][:])
    cfg.scene.num_envs=num_envs;cfg.seed=seed;cfg.sim.device='cuda:0';cfg.sim.gravity=(0,0,-9.81)
    cfg.log_dir=str(output)
    cfg.scene.robot.spawn.usd_dir=str(output/'robot_usd')
    cfg.scene.robot.spawn.usd_file_name='reduced28.usd'
    cfg.scene.robot.init_state.pos=tuple(root);cfg.scene.robot.init_state.rot=tuple(quat)
    # Exact names replace regex defaults; overlapping regex/exact entries are rejected by IsaacLab.
    import xml.etree.ElementTree as ET
    initial={**arm,**hand}
    for joint in ET.parse(ASSEMBLY_URDF_PATH).getroot().findall('joint'):
        limit=joint.find('limit');name=joint.get('name')
        if name in initial and limit is not None and 'lower' in limit.attrib and 'upper' in limit.attrib:
            initial[name]=float(np.clip(initial[name],float(limit.get('lower'))+1e-6,
                                       float(limit.get('upper'))-1e-6))
    cfg.scene.robot.init_state.joint_pos=initial
    scene=json.loads((inputs/'scene_from_object.json').read_text())['table']
    cfg.scene.table.init_state.pos=tuple(scene['center_m']);cfg.scene.table.spawn.size=tuple(scene['size_m'])
    motion=cfg.commands.motion
    motion.demo_path=motion.retargeted_traj_path=str(inputs/REFERENCE);motion.demo_dt=1/120
    motion.use_reference_trajectory_as_demo=True
    motion.arm_joint_trajectory_path=str(inputs/SIDECAR)
    motion.arm_joint_trajectory_urdf_path=ASSEMBLY_URDF_PATH
    motion.arm_joint_trajectory_reference_sha256=info['reference_sha256']
    motion.arm_reset_precomputed_bank_path=None;motion.arm_reset_augmented_ik_enabled=False
    motion.augmentation_traj_dir=None;motion.traj_aug_enabled=False
    motion.enable_reset_perturbation=False;motion.adaptive_sampling_enabled=False
    motion.object_pos_lower_bound=tuple(lower);motion.object_pos_upper_bound=tuple(upper)
    cfg.actions.root_pose.gravity_compensation_scale_range=(1.,1.)
    for name in ('robot_physics_material','robot_scale_mass','robot_joint_stiffness_and_damping',
        'init_obs_delay_buffers','table_physics_material','object_physics_material','object_scale_mass',
        'object_com','object_scale','observation_time_lag','curriculum_random_push_robot',
        'curriculum_random_push_object','curriculum_gravity'):
        if hasattr(cfg.events,name):setattr(cfg.events,name,None)
    for group in (cfg.observations.policy,cfg.observations.critic):
        group.enable_corruption=False
        for term in vars(group).values():
            params=getattr(term,'params',None)
            if isinstance(params,dict) and 'apply_noise' in params:params['apply_noise']=False
    cfg.obs_max_lags={key:0 for key in cfg.obs_max_lags}
    for i in range(1,6):
        sensor=getattr(cfg.scene,f'contact_fingertip{i}_object',None)
        if sensor is not None and hasattr(sensor,'track_contact_points'):
            sensor.track_contact_points=True
            if hasattr(sensor,'max_contact_data_count_per_prim'):
                sensor.max_contact_data_count_per_prim=64
    (output/'task_adaptation.json').write_text(json.dumps(dict(input=info,
        baseline_commit='5c409994e4e04fd131aa46ad8441480da025c7fd',robot_joint_count=28,
        ZIP_only=True,original_bank_disabled_for_new_reference=True,
        control_reward_architecture='requested branch, unchanged',nominal_no_DR=True),indent=2))
    return cfg


def apply(cfg,args,output):
    cfg=configure(cfg,Path(args.trajectory).parent,output,args.num_envs,args.seed)
    cfg.commands.motion.rsi_enabled=False
    return cfg


def runner_for(env, task, seed, output=None):
    import rsl_rl.runners.on_policy_runner as module
    from rsl_rl.runners import OnPolicyRunner
    from regrind.modules.actor_critic import CustomActorCritic
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    module.CustomActorCritic=CustomActorCritic
    cfg=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point');cfg.seed=seed;cfg.device='cuda:0'
    wrapped=RslRlVecEnvWrapper(env,clip_actions=cfg.clip_actions)
    runner=OnPolicyRunner(wrapped,cfg.to_dict(),log_dir=str(output) if output else None,device='cuda:0')
    return runner,wrapped


def policy_for(env,args):
    checkpoint=os.environ.get('C2DEX_POLICY_CHECKPOINT')
    if not checkpoint:return None
    runner,_=runner_for(env,args.task,args.seed)
    runner.load(checkpoint,load_optimizer=False,map_location='cuda:0')
    return runner.get_inference_policy(device='cuda:0')
