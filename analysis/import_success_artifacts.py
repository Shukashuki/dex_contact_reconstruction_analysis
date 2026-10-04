"""Export only admitted successful-pickup derivatives from a local dexpipe run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from h5py import File as h5py_file
from eval.screwdriver_rotation_compare import principal_axis


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dexpipe',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    a=p.parse_args();source=a.dexpipe
    experiment=source/'rl/output/c2dex_front_table_20261003_v1'
    returns=source/'data/server_returns/c2dex_front_table_20261003_v1'
    case=returns/'baseline28_tuned'
    state=json.loads((case/'baseline28_tuned.evaluation.status.json').read_text())
    summary=json.loads((case/'summary.json').read_text());trial=summary['trials'][0]
    threshold=summary['thresholds']
    if state['status']!='completed' or not trial['physics_valid'] or not trial['initial_state_valid'] or trial['silent_reset_count']:
        raise ValueError('Only completed uninterrupted valid physical rollouts can be exported')
    if trial['max_relative_lift_m']<threshold['lift_height_m'] or trial['max_hold_seconds']<threshold['hold_seconds']:
        raise ValueError('The selected case did not achieve lift and hold')
    destination=a.output/'screwdriver/successful_ppo';destination.mkdir(parents=True,exist_ok=True)
    clean={k:summary[k] for k in ('task','algorithm','seed','num_trials','step_dt_s','frames_replayed',
        'reference_frames','scored_reference_frames','evaluation_scope','thresholds','trials')}
    clean['evaluation_status']=state['status']
    collection=json.loads((returns/'collection_manifest.json').read_text())
    checkpoint=collection['checkpoints']['baseline']['sha256']
    clean['evaluation_contract']={k:v for k,v in summary['evaluation_contract'].items()
        if k not in ('arm_ik_trajectory','policy_checkpoint')}
    clean['evaluation_contract']['policy_checkpoint']='sha256:'+checkpoint
    (destination/'summary.json').write_text(json.dumps(clean,indent=2))
    with np.load(case/'assembly_trace.npz',allow_pickle=False) as trace:
        keys=('object_pos','object_quat','contact_pos_w','fingertip_force_n',
              'reference_object_pos','reference_object_quat','fps')
        np.savez_compressed(destination/'assembly_trace.npz',**{k:trace[k] for k in keys})
    with np.load(experiment/'offline_contact_trace.npz') as targets:
        np.savez_compressed(a.output/'screwdriver/predictions.npz',
            **{k:targets[k] for k in ('frame','finger','target_object')})
    original=json.loads((experiment/'manifest.json').read_text())
    robot=json.loads((experiment/'reduced28_v2/baseline/manifest.json').read_text())
    training=json.loads((returns/'training_baseline/training_summary.json').read_text())
    public_training={k:training[k] for k in ('completed','initial_iteration','final_iteration',
        'additional_updates','environments','rollout_steps','transitions','wall_seconds','scope')}
    public_training.update(initial_checkpoint='model_5200.pt',final_checkpoint_sha256=checkpoint)
    (a.output/'screwdriver/training.json').write_text(json.dumps(public_training,indent=2))
    metadata=dict(case='front table screwdriver ZIP',robot_dof=28,seed=42,
        baseline_commit='5c409994e4e04fd131aa46ad8441480da025c7fd',
        original_checkpoint_sha256='87fa35f2ea6af757c35887b0fd59bb496fd17099a73dc0649815cb9ccba4d278',
        finetuned_checkpoint_sha256=checkpoint,robot_sha256=robot['robot_sha256'],
        archive_sha256=original['archive_sha256'],object_usd_sha256=original['object_usd_sha256'],
        canonical_mesh_axis=principal_axis(experiment/'screwdriver_object_m.obj').tolist(),
        original_trace_sha256=digest(case/'assembly_trace.npz'),
        published_trace_sha256=digest(destination/'assembly_trace.npz'),
        predictions_sha256=digest(a.output/'screwdriver/predictions.npz'),
        summary_sha256=digest(destination/'summary.json'),
        success_definition='lift >=5 cm and hold >3 cm for >=0.5 s; full rotation task success is separate',
        predicted_labels='C2Dex-style 3D proximity contacts; no camera rays in the screwdriver ZIP',
        checkpoint_lineage='original checkpoint used a different demo; only additional fine-tuning is ZIP-only',
        case_condition='baseline without contact-preserving C2Dex retarget; no matched reconstructed successful pickup',
        data_policy='derived contact/pose arrays only; no MANO weights, source archive, robot meshes or checkpoints')
    (a.output/'screwdriver/metadata.json').write_text(json.dumps(metadata,indent=2))
    with h5py_file(experiment/'baseline/reference_contact_gap_tron2_front_final_120fps.h5') as h:
        joint_names=[x.decode() if isinstance(x,bytes) else str(x) for x in h['meta/joint_names'][:]]
    with np.load(experiment/'robot_model.npz',allow_pickle=False) as model:
        mapping=dict(joint_names=joint_names,joint_limits=model['limits'].tolist(),
            node_body_names=[str(model['body_names'][i]) for i in model['nodes']],
            human_ids=model['human_ids'].tolist(),wrist_to_handroot_rotation=model['C'].tolist())
    (a.output/'revo3_contract.json').write_text(json.dumps(mapping,indent=2))
    for label,compare_name,replay_name in (
        ('taco_press','c2dex_sim_contact_compare_success_20231013_304_left.json','c2dex_contact_replay_success_20231013_304_left.json'),
        ('taco_stir_fry','c2dex_sim_contact_compare_success_stir-fry__20231104_156_right.json','c2dex_contact_replay_success_stir-fry__20231104_156_right.json')):
        replay_path=source/'retarget/output'/replay_name
        comparison_path=source/'eval/output'/compare_name
        replay=json.loads(replay_path.read_text())
        if not all(replay[k] for k in ('ref_lifts','obj_lifted','gate_criterion_met')) or replay['held_aloft_frames']<=0:
            raise ValueError('Supplementary case must pass its original project physics gate')
        report=json.loads(comparison_path.read_text())
        report['physics']={k:replay[k] for k in ('ref_lifts','obj_lifted','gate_criterion_met',
            'sim_lift_m','held_aloft_frames','mode','init_frame')}
        report['source_comparison_sha256']=digest(comparison_path)
        report['source_physics_sha256']=digest(replay_path)
        folder=a.output/'taco_success';folder.mkdir(exist_ok=True)
        (folder/(label+'.json')).write_text(json.dumps(report,indent=2))
    print('Exported one screwdriver lift/hold case and two project-gate-passing TACO cases; no failed experimental outputs.')


if __name__=='__main__':main()
