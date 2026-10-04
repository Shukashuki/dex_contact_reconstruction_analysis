import copy
import json
from pathlib import Path
import numpy as np
import pytest
from analysis.successful_screwdriver import eligible
from analysis.metrics import analyze

ROOT=Path(__file__).resolve().parents[1]


def summary():
    return json.loads((ROOT/'data/screwdriver/successful_ppo/summary.json').read_text())


def test_successful_pickup_is_not_full_rotation_success():
    data=summary()
    assert eligible(data)
    assert not data['trials'][0]['success']
    assert data['trials'][0]['failure_reasons']==['rotation_rmse_exceeded','rotation_max_exceeded']


@pytest.mark.parametrize('field,value',[
    ('max_relative_lift_m',0),('max_hold_seconds',0),('physics_valid',False),
    ('initial_state_valid',False),('silent_reset_count',1),('reached_reference_end',False)])
def test_failed_pickup_or_invalid_physics_is_not_admitted(field,value):
    data=copy.deepcopy(summary());data['trials'][0][field]=value
    assert not eligible(data)


def test_published_contact_statistics_and_missing_labels():
    data=ROOT/'data/screwdriver'
    metadata=json.loads((data/'metadata.json').read_text())
    with np.load(data/'predictions.npz',allow_pickle=False) as d:targets={k:d[k] for k in d.files}
    result,_=analyze(data/'successful_ppo',targets,np.asarray(metadata['canonical_mesh_axis']))
    held=result['held_spatial_region_comparison']
    assert result['held_contact_frames']==209
    assert held['actual_held_centroids']==534 and held['compared_centroids']==372
    assert held['nearest_region_mean_mm']==pytest.approx(67.26899707962481)
    assert held['fraction_within_20mm']==pytest.approx(42/372)
    assert result['held_phase_stable_contact']['predicted_samples']==0
    assert result['held_phase_stable_contact']['centroid_distance_mean_mm'] is None
    assert not held['per_finger']['middle']['predicted_region_available']


def test_supplementary_taco_cases_passed_their_own_gate():
    for file in (ROOT/'data/taco_success').glob('*.json'):
        physics=json.loads(file.read_text())['physics']
        assert physics['gate_criterion_met'] and physics['ref_lifts'] and physics['obj_lifted']
        assert physics['held_aloft_frames']>0
