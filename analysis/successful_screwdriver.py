"""Recompute contacts for the admitted successful pickup, excluding failed cases."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from analysis.metrics import analyze


def eligible(summary):
    t=summary['trials'][0];limits=summary['thresholds']
    return (summary.get('evaluation_status')=='completed' and t['initial_state_valid']
        and t['physics_valid'] and t['reached_reference_end'] and t['silent_reset_count']==0
        and t['max_relative_lift_m']>=limits['lift_height_m']
        and t['max_hold_seconds']>=limits['hold_seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data/screwdriver')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/screwdriver_success.json')
    a=p.parse_args();metadata=json.loads((a.data/'metadata.json').read_text())
    case=a.data/'successful_ppo';summary=json.loads((case/'summary.json').read_text())
    if not eligible(summary):raise ValueError('Failed pickup cases are excluded from this analysis')
    for path,key in ((case/'assembly_trace.npz','published_trace_sha256'),
        (case/'summary.json','summary_sha256'),(a.data/'predictions.npz','predictions_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata[key]:raise ValueError('Published input hash differs')
    with np.load(a.data/'predictions.npz',allow_pickle=False) as d:targets={k:d[k] for k in d.files}
    result,trace=analyze(case,targets,np.asarray(metadata['canonical_mesh_axis']))
    result['label']='Successful screwdriver pickup after baseline PPO adaptation'
    report=dict(scope='success-conditioned physical pickup contact analysis',successful_case=result,
        reconstruction_effect=dict(status='not_established',matched_successful_pairs=0,
            explanation='No matched successful pickup with and without reconstruction; successful baseline is not evidence of reconstruction benefit'),
        training=json.loads((a.data/'training.json').read_text()),provenance=metadata,
        limitations=['Contact labels cover only 9/270 frames, all before the held phase',
            'Held contact distance to earlier regions is time-agnostic, NOT held-phase prediction accuracy',
            'Same-finger sensor centroids are not all raw narrowphase contact points',
            'Middle-finger held contacts have no predicted region and are excluded from distance statistics',
            'Lift and hold passed; full rotation tracking failed',
            'Single seed, bounded warm-start adaptation; not full C2Dex or original PPO training reproduction',
            'Success-conditioned subset cannot estimate overall method success rate'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    np.savez_compressed(a.output.with_suffix('.contacts.npz'),
        predicted_frame=targets['frame'],predicted_finger=targets['finger'],
        predicted_contact_object=targets['target_object'],
        actual_contact_object=trace['canonical'],actual_contact_valid=trace['valid'])
    plot(case,trace,targets,a.output.with_suffix('.png'))
    print(json.dumps(dict(lift_cm=result['physics']['max_relative_lift_m']*100,
        hold_s=result['physics']['max_hold_seconds'],held_contact_frames=result['held_contact_frames'],
        held_spatial_comparison=result['held_spatial_region_comparison'],
        reconstruction_improvement='not established'),indent=2))


def plot(case,trace,targets,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(case/'assembly_trace.npz',allow_pickle=False) as d:pos=d['object_pos'][:,0]
    lift=(pos[:,2]-pos[0,2])*100;held=lift>3
    fig=plt.figure(figsize=(12,9));axes=[fig.add_subplot(221),fig.add_subplot(222),
        fig.add_subplot(223,projection='3d'),fig.add_subplot(224)]
    axes[0].plot(trace['time'],lift,label='Simulated object lift')
    axes[0].axhline(5,color='gray',linestyle='--',label='Lift threshold 5 cm')
    axes[0].axhline(3,color='gray',linestyle=':',label='Hold height 3 cm')
    axes[0].set_ylabel('Relative object lift (cm)');axes[0].set_xlabel('Time (s)');axes[0].legend(fontsize=8)
    axes[1].plot(trace['time'],trace['contacts'],label='Object-contacting fingertips')
    start,end=targets['frame'].min()/30,targets['frame'].max()/30
    for ax in axes[:2]:ax.axvspan(start,end,color='orange',alpha=.2,label='Prediction label interval')
    axes[1].set_ylabel('Fingertips with >0.1 N contact');axes[1].set_xlabel('Time (s)');axes[1].legend(fontsize=8)
    means=[];names=('thumb','index','middle','ring','little')
    for finger,name in enumerate(names):
        points=trace['canonical'][held&trace['valid'][:,finger],finger]
        if len(points):axes[2].scatter(*points.T*1000,s=8,alpha=.6,label=name)
        predicted=targets['target_object'][targets['finger']==finger]
        values=cKDTree(predicted).query(points)[0]*1000 if len(predicted) else np.array([])
        means.append(float(values.mean()) if len(values) else np.nan)
    axes[2].scatter(*targets['target_object'].T*1000,color='black',marker='x',label='Earlier predicted regions')
    axes[2].set_xlabel('Object x (mm)');axes[2].set_ylabel('Object y (mm)');axes[2].set_zlabel('Object z (mm)')
    axes[2].set_title('Actual held contacts and earlier reconstruction');axes[2].legend(fontsize=7)
    axes[3].bar(names,means,color='steelblue')
    for i,value in enumerate(means):
        if np.isfinite(value):axes[3].text(i,value+2,f'{value:.1f}',ha='center')
        else:axes[3].text(i,3,'No predicted\nregion',ha='center',fontsize=8)
    axes[3].set_ylabel('Mean nearest region distance (mm)');axes[3].set_ylim(0,120)
    axes[3].set_title('Time-agnostic same-finger comparison')
    fig.suptitle('Successful pickup after PPO is not successful screwdriver rotation')
    fig.tight_layout();fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':main()
