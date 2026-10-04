"""Compare measured PhysX fingertip contact centroids, not collision proxies."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from eval.screwdriver_rotation_compare import principal_axis, twist


def analyze(directory, targets, axis):
    summary=json.loads((directory/'summary.json').read_text())
    trial=summary['trials'][0]
    if not trial['physics_valid'] or trial['silent_reset_count']:
        raise ValueError('Only valid uninterrupted physical rollouts are comparable')
    with np.load(directory/'assembly_trace.npz',allow_pickle=True) as d:
        pos=d['object_pos'][:,0];quat=d['object_quat'][:,0]
        refpos=d['reference_object_pos'];refquat=d['reference_object_quat']
        contact=d['contact_pos_w'][:,0];force=d['fingertip_force_n'][:,0]
        fps=float(d['fps'])
    if contact.shape!=(len(pos),5,3) or force.shape!=(len(pos),5) or fps<=0:
        raise ValueError('Expected exactly one object-filtered centroid for each of five fingertips')
    actual=Rotation.from_quat(quat[:,[1,2,3,0]])
    reference=Rotation.from_quat(refquat[:,[1,2,3,0]])
    ra,rd=twist(reference,reference[0],axis);sa,sd=twist(actual,reference[0],axis)
    active=(np.abs(np.degrees(np.gradient(ra))*fps)>=10)&rd&sd
    valid=(force>.1)&np.isfinite(contact).all(-1)
    canonical=np.einsum('tji,tfj->tfi',actual.as_matrix(),contact-pos[:,None,:])
    frames=targets['frame'].astype(int);fingers=targets['finger'].astype(int)
    if np.any(frames<0) or np.any(frames>=len(pos)) or np.any(fingers<0) or np.any(fingers>=5):
        raise ValueError('Contact prediction does not align with the physical trace')
    observed=valid[frames,fingers]
    distance=np.linalg.norm(canonical[frames,fingers]-targets['target_object'],axis=-1)*1000
    def contact_metrics(mask):
        selected=mask&observed;values=distance[selected]
        pairs=np.unique(np.stack([frames[mask],fingers[mask]],axis=1),axis=0)
        region_distances=[];measured_frames=[]
        for frame,finger in pairs:
            if valid[frame,finger]:
                members=mask&(frames==frame)&(fingers==finger)
                region_distances.append(float(distance[members].min()))
                measured_frames.append(int(frame))
        return dict(predicted_samples=int(mask.sum()),measured_samples=int(selected.sum()),
            predicted_frames=int(len(np.unique(frames[mask]))),measured_frames=len(set(measured_frames)),
            measured_coverage=float(selected.sum()/mask.sum()) if mask.any() else None,
            predicted_finger_frames=len(pairs),measured_finger_frames=len(region_distances),
            finger_frame_coverage=len(region_distances)/len(pairs) if len(pairs) else None,
            nearest_predicted_region_mean_mm=float(np.mean(region_distances)) if region_distances else None,
            distance_weighting='all stable vertex constraints; region metric gives each finger/frame equal weight',
            centroid_distance_mean_mm=float(values.mean()) if len(values) else None,
            centroid_distance_p90_mm=float(np.percentile(values,90)) if len(values) else None)
    count=valid.sum(-1);held=pos[:,2]-pos[0,2]>.03
    held_distances=[];held_by_finger={}
    finger_names=('thumb','index','middle','ring','little')
    for finger,name in enumerate(finger_names):
        selected=held&valid[:,finger]
        predicted=targets['target_object'][fingers==finger]
        values=cKDTree(predicted).query(canonical[selected,finger])[0]*1000 if len(predicted) else np.array([])
        held_distances.extend(values.tolist())
        held_by_finger[name]=dict(actual_held_centroids=int(selected.sum()),
            predicted_region_available=bool(len(predicted)),
            nearest_region_mean_mm=float(values.mean()) if len(values) else None)
    held_spatial=dict(scope='time-agnostic spatial comparison to the union of earlier predicted same-finger regions; NOT held-phase prediction accuracy',
        actual_held_centroids=int((valid&held[:,None]).sum()),compared_centroids=len(held_distances),
        nearest_region_mean_mm=float(np.mean(held_distances)) if held_distances else None,
        nearest_region_p90_mm=float(np.percentile(held_distances,90)) if held_distances else None,
        fraction_within_20mm=float(np.mean(np.asarray(held_distances)<=20)) if held_distances else None,
        per_finger=held_by_finger)
    runs=[];length=0
    for value in count>=3:
        length=length+1 if value else 0;runs.append(length)
    result=dict(label=directory.name,physical_success=trial['success'],physics=trial,
        variant='c2dex' if directory.name.startswith('c2dex') else 'ZIP_baseline',
        stage='zero_residual' if '_zero' in directory.name else 'pretrained_5200' if '_pretrained' in directory.name else 'PPO_finetuned_200',
        lift_and_hold_achieved=trial.get('max_relative_lift_m',0)>=summary.get('thresholds',{}).get('lift_height_m',.05)
            and trial.get('max_hold_seconds',0)>=summary.get('thresholds',{}).get('hold_seconds',.5),
        policy_checkpoint=summary['evaluation_contract']['policy_checkpoint'],
        contact_measurement='object-filtered PhysX per-fingertip contact centroids; >0.1 N, finite',
        stable_contact=contact_metrics(np.ones(len(frames),bool)),
        rotation_active_stable_contact=contact_metrics(active[frames]),
        held_phase_stable_contact=contact_metrics(held[frames]),
        held_spatial_region_comparison=held_spatial,
        contact_frames=int((count>0).sum()),rotation_active_contact_frames=int(((count>0)&active).sum()),
        max_finger_contacts=int(count.max()),three_finger_frames=int((count>=3).sum()),
        measured_contact_frames_per_finger=dict(zip(('thumb','index','middle','ring','little'),
                                                   map(int,valid.sum(0)))),
        longest_three_finger_s=max(runs)/fps,held_contact_frames=int((held&(count>0)).sum()),
        axial_net_deg=float(np.degrees(sa[-1]-sa[0])),axial_travel_deg=float(np.degrees(np.abs(np.diff(sa)).sum())),
        reference_axial_net_deg=float(np.degrees(ra[-1]-ra[0])),
        twist_undefined_frames=int((~(rd&sd)).sum()),
        axial_rmse_deg=float(np.sqrt(np.mean(np.degrees((sa-ra)[rd&sd])**2))) if (rd&sd).any() else None,
        rotation_active_axial_rmse_deg=float(np.sqrt(np.mean(np.degrees(sa[active]-ra[active])**2))) if active.any() else None,
        trace_sha256=hashlib.sha256((directory/'assembly_trace.npz').read_bytes()).hexdigest())
    return result,dict(time=np.arange(len(pos))/fps,ref=np.degrees(ra),actual=np.degrees(sa),
        contacts=count,canonical=canonical,valid=valid,reference_position=refpos,reference_quaternion=refquat)
