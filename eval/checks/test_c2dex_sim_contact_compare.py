from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.c2dex_sim_contact_compare import compare, unique_points  # noqa: E402


def test_unique_points_removes_substep_duplicates() -> None:
    points = np.array([[0, 0, 0], [.0005, 0, 0], [.01, 0, 0]])
    assert len(unique_points(points, .001)) == 2


def test_compare_uses_same_frame_and_both_directions() -> None:
    predicted = [np.array([[0., 0., 0.]]), np.empty((0, 3)),
                 np.array([[.02, 0., 0.]])]
    simulated = [np.array([[.003, 0., 0.]]), np.array([[0., 0., 0.]]),
                 np.array([[.04, 0., 0.]])]
    result = compare(predicted, simulated)
    assert result["overlap_frames"] == 2
    assert result["temporal_iou"] == 2 / 3
    assert result["prediction_to_sim"]["median_mm"] == 11.5
    assert result["thresholds"]["5mm"]["prediction_coverage"] == .5
