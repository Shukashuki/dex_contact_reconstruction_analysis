"""Portable constants needed by the user-selected 28-DOF PPO adapter."""
import hashlib
from pathlib import Path
import numpy as np

REFERENCE = 'reference_contact_gap_tron2_front_final_120fps.h5'
SIDECAR = 'arm_ik_tron2_front_final_270f.npz'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workspace_bounds(positions):
    positions=np.asarray(positions,dtype=float)
    if positions.ndim!=2 or positions.shape[1]!=3 or not len(positions) or not np.isfinite(positions).all():
        raise ValueError('Invalid reference positions')
    center=(positions.min(axis=0)+positions.max(axis=0))/2
    return ([float(center[0]-.25),float(center[1]-.25),.5],
            [float(center[0]+.25),float(center[1]+.25),1.])
