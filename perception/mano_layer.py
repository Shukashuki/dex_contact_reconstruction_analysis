"""Pure-numpy MANO forward kinematics (no torch, no chumpy).

Datasets such as TACO ship MANO *parameters*; every downstream consumer in
dexpipe wants 21 joint positions (`HandObjectTraj`, `joints.hdf5`).  This
module closes that gap without dragging torch into the main venv, which is a
standing constraint of this pipeline.

The official `MANO_*.pkl` files are chumpy pickles.  chumpy does not install
cleanly on modern numpy, so the model is unpickled with a stub that stands in
for every chumpy class and the underlying arrays are pulled out afterwards --
`shapedirs` in particular arrives as a lazily-reordered view over a (778, 3,
20) buffer and is materialised here.

Verification is by reprojection, not by trusting the constants: see
`perception/checks/check_mano_reprojection.py`.  Fingertip vertex indices and
the `flat_hand_mean` convention are exactly the kind of thing that silently
shifts a hand by centimetres, and this repository has been bitten three times
by unverified frame constants.
"""
from __future__ import annotations

import pickle
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MANO_ROOT = Path(os.environ.get("MANO_ROOT", "private_inputs/mano/models"))

# MANO regresses 16 joints; the five fingertips are taken from mesh vertices.
# These are the indices used by manopth/manotorch, whose 21-joint "SNAP"
# ordering the rest of this pipeline follows.
FINGERTIP_VERTICES = {
    "thumb": 745, "index": 317, "middle": 444, "ring": 556, "pinky": 673,
}
# MANO joint order -> SNAP 21-joint order (wrist, then thumb..pinky, 4 each).
# MANO: 0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb.
SNAP_FROM_MANO = [0, 13, 14, 15, None,      # wrist, thumb + tip
                  1, 2, 3, None,            # index
                  4, 5, 6, None,            # middle
                  10, 11, 12, None,         # ring
                  7, 8, 9, None]            # pinky
SNAP_TIP_SLOTS = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}

JOINT_NAMES = [
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


class _ChumpyStub:
    """Placeholder for any chumpy class encountered while unpickling."""

    def __init__(self, *args, **kwargs) -> None:
        self._args = args

    def __setstate__(self, state) -> None:
        self.__dict__.update(state if isinstance(state, dict)
                             else {"_state": state})

    def __call__(self, *args, **kwargs) -> "_ChumpyStub":
        return _ChumpyStub(*args)


class _ManoUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("chumpy"):
            return _ChumpyStub
        return super().find_class(module, name)


def _materialise(value):
    """Turn a chumpy stub back into the ndarray it was standing in for."""
    if isinstance(value, np.ndarray):
        return value
    if not isinstance(value, _ChumpyStub):
        return value
    state = value.__dict__
    for key in ("x", "_data", "data"):
        if key in state:
            return np.asarray(state[key])
    # A reordering view: gather from the wrapped buffer, then reshape.
    if "a" in state and "idxs" in state:
        base = np.asarray(_materialise(state["a"]))
        gathered = base.ravel()[np.asarray(state["idxs"])]
        shape = state.get("preferred_shape")
        return gathered.reshape(shape) if shape else gathered
    raise TypeError(f"cannot materialise chumpy object with keys {list(state)}")


@dataclass(frozen=True)
class ManoModel:
    """The subset of MANO needed for joints and fingertips."""

    side: str
    v_template: np.ndarray      # (778, 3)
    shapedirs: np.ndarray       # (778, 3, 10)
    posedirs: np.ndarray        # (778, 3, 135)
    J_regressor: np.ndarray     # (16, 778) dense
    weights: np.ndarray         # (778, 16)
    parents: np.ndarray         # (16,) int, parents[0] == -1
    hands_mean: np.ndarray      # (45,)
    faces: np.ndarray           # (1538, 3)


def load_mano(side: str, root: Path = MANO_ROOT) -> ManoModel:
    path = Path(root) / f"MANO_{side.upper()}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. MANO models are licensed separately; this "
            f"pipeline reads the copy under {root}.")
    with open(path, "rb") as handle:
        raw = _ManoUnpickler(handle, encoding="latin1").load()
    regressor = raw["J_regressor"]
    parents = np.asarray(raw["kintree_table"])[0].astype(np.int64).copy()
    parents[0] = -1
    return ManoModel(
        side=side.lower(),
        v_template=np.asarray(_materialise(raw["v_template"]), dtype=np.float64),
        shapedirs=np.asarray(_materialise(raw["shapedirs"]), dtype=np.float64),
        posedirs=np.asarray(_materialise(raw["posedirs"]), dtype=np.float64),
        J_regressor=np.asarray(regressor.todense() if hasattr(regressor, "todense")
                               else regressor, dtype=np.float64),
        weights=np.asarray(_materialise(raw["weights"]), dtype=np.float64),
        parents=parents,
        hands_mean=np.asarray(_materialise(raw["hands_mean"]), dtype=np.float64),
        faces=np.asarray(raw["f"], dtype=np.int64),
    )


def axis_angle_to_matrix(vectors: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrices (Rodrigues)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    angle = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = np.where(angle < 1e-12, 1.0, angle)
    axis = vectors / safe
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack([zero, -z, y, z, zero, -x, -y, x, zero],
                    axis=-1).reshape(*axis.shape[:-1], 3, 3)
    eye = np.broadcast_to(np.eye(3), skew.shape).copy()
    sin = np.sin(angle)[..., None]
    cos = np.cos(angle)[..., None]
    rotation = eye + sin * skew + (1 - cos) * (skew @ skew)
    return np.where((angle[..., None] < 1e-12), eye, rotation)


def forward(model: ManoModel, pose: np.ndarray, betas: np.ndarray,
            trans: np.ndarray | None = None, *,
            flat_hand_mean: bool = False,
            center_idx: int | None = 0
            ) -> tuple[np.ndarray, np.ndarray]:
    """MANO forward pass.

    pose  : (T, 48) axis-angle -- 3 global orientation + 45 finger DOF
    betas : (10,) shape coefficients
    trans : (T, 3) global translation, or None
    flat_hand_mean : False adds ``hands_mean`` to the finger DOF, which is
        what manopth/manotorch do by default and what TACO's parameters
        assume.  Getting this wrong curls the whole hand.
    center_idx : joint moved to the origin before ``trans`` is applied.  0
        (the wrist) matches manopth/manotorch and is what TACO's
        ``hand_trans`` means -- verified by reprojection: with centring the
        wrist lands on the imaged wrist, without it the whole hand sits a
        template offset away.  None disables centring.

    Returns (joints21 (T, 21, 3), vertices (T, 778, 3)).
    """
    pose = np.atleast_2d(np.asarray(pose, dtype=np.float64))
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    if pose.shape[1] != 48:
        raise ValueError(f"pose must be (T, 48), got {pose.shape}")
    if betas.shape != (10,):
        raise ValueError(f"betas must be (10,), got {betas.shape}")
    frames = len(pose)

    full = pose.copy()
    if not flat_hand_mean:
        full[:, 3:] = full[:, 3:] + model.hands_mean

    v_shaped = model.v_template + model.shapedirs @ betas          # (778, 3)
    rest_joints = model.J_regressor @ v_shaped                     # (16, 3)

    rotations = axis_angle_to_matrix(full.reshape(frames, 16, 3))  # (T,16,3,3)

    # Pose-corrective blend shapes use the 15 non-root rotations.
    pose_feature = (rotations[:, 1:] - np.eye(3)).reshape(frames, 135)
    v_posed = v_shaped + np.einsum("vdp,tp->tvd", model.posedirs, pose_feature)

    # Forward kinematics along the kinematic tree.
    absolute = np.zeros((frames, 16, 4, 4))
    absolute[:, 0, :3, :3] = rotations[:, 0]
    absolute[:, 0, :3, 3] = rest_joints[0]
    absolute[:, 0, 3, 3] = 1.0
    for joint in range(1, 16):
        parent = model.parents[joint]
        local = np.zeros((frames, 4, 4))
        local[:, :3, :3] = rotations[:, joint]
        local[:, :3, 3] = rest_joints[joint] - rest_joints[parent]
        local[:, 3, 3] = 1.0
        absolute[:, joint] = absolute[:, parent] @ local
    joints16 = absolute[:, :, :3, 3]

    # Skinning: strip the rest pose out of each bone transform first.
    rest_h = np.concatenate([rest_joints, np.zeros((16, 1))], axis=1)
    offset = np.zeros((frames, 16, 4, 4))
    offset[..., 3] = np.einsum("tjab,jb->tja", absolute, rest_h)
    relative = absolute - offset
    skinned = np.einsum("vj,tjab->tvab", model.weights, relative)
    v_homo = np.concatenate(
        [v_posed, np.ones((frames, v_posed.shape[1], 1))], axis=2)
    vertices = np.einsum("tvab,tvb->tva", skinned, v_homo)[..., :3]

    joints = np.zeros((frames, 21, 3))
    for slot, mano_index in enumerate(SNAP_FROM_MANO):
        if mano_index is not None:
            joints[:, slot] = joints16[:, mano_index]
    for slot, finger in SNAP_TIP_SLOTS.items():
        joints[:, slot] = vertices[:, FINGERTIP_VERTICES[finger]]

    if center_idx is not None:
        origin = joints[:, center_idx:center_idx + 1]
        joints = joints - origin
        vertices = vertices - origin
    if trans is not None:
        trans = np.asarray(trans, dtype=np.float64).reshape(frames, 1, 3)
        joints = joints + trans
        vertices = vertices + trans
    return joints, vertices
