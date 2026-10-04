# Reproducing Contact Reconstruction and Successful Case Analysis

The analysis can be rerun directly. Full reconstruction and new PPO training require locally obtained raw data, MANO, a MuJoCo robot, the specified dex-rl branch, and an Isaac environment. These assets are not bundled in this repository.

## Rerun the Published Successful Case

```bash
python -m pip install -e '.[test]'
python -m analysis.successful_screwdriver
python -m pytest -q
```

The entry point checks pickup subcriteria and input SHA256 hashes, then recomputes object-coordinate contacts, synchronized label coverage, held-phase region distances, and the plot. Without a same-finger predicted region, distance is null rather than zero. Outputs are `results/screwdriver_success.json`, a PNG, and a contact NPZ.

## Local MANO Assets

```bash
python -m pip install -e '.[reconstruction,robot,test]'
python -m perception.export_mano \
  --root private_inputs/mano/models --side right \
  --output perception/hand_artifacts/mano_right.npz
```

Load pickle files only from trusted, legally obtained sources. Use `MANO_ROOT` for the pickle directory and `MANO_NPZ_ROOT` for the exported NPZ directory. Do not commit weights to this repository.

## Screwdriver Reconstruction Inputs

Prepare the original ZIP, the corresponding canonical object OBJ in meters, and table metadata. The original ZIP SHA256 is `4e2735400b5e44a28ac6f05bda926693ed2944c6dec787b1f7c8b0c32f8f92cb`; other files must not be substituted as this experiment's input.

```bash
python -m perception.screwdriver_fit --prepare \
  --root private_inputs/screwdriver \
  --archive private_inputs/front_table_screwdriver.zip \
  --mesh private_inputs/screwdriver_object_m.obj \
  --scene data/screwdriver/scene_from_object.json
python -m perception.screwdriver_fit --root private_inputs/screwdriver
```

Preparation extracts only HDF and sidecar data, preserving object and MANO references without extracting video. Fitting uses zero MANO betas, a bone-length least-squares scale, 3D keypoint fitting, 8 mm proximity, and an opposite-normal score above 0.5, followed by contact aggregation over stable segments and the dominant cluster. The recorded scale is 1.65208, with fit mean 2.839 mm, median 1.705 mm, and P90 8.049 mm. These are keypoint-fitting measurements, not simulated contact errors.

`perception/c2dex_reconstruction.py` also provides ray extraction when camera inputs are available, and `perception/c2dex_optimize.py` provides MANO trajectory optimization. Labels for this successful screwdriver case come only from the 3D fitting and proximity path above. Providing these modules does not establish that this case ran silhouette-ray reconstruction or reconstruction trajectory optimization.

## Contact Preserving Retargeting

Export a model from a legally obtained local fixed-base Revo3 MuJoCo XML. `data/revo3_contract.json` explicitly specifies joints, keypoints, and wrist-to-handroot rotation; coordinate transforms are not inferred.

```bash
python -m retarget.export_model \
  --xml private_inputs/revo3/revo3_right.xml \
  --contract data/revo3_contract.json \
  --output private_inputs/screwdriver/robot_model.npz
python -m retarget.c2dex_retarget \
  --model private_inputs/screwdriver/robot_model.npz \
  --human private_inputs/screwdriver/human.npz \
  --contacts private_inputs/screwdriver/stable_contacts.npz \
  --template private_inputs/screwdriver/baseline_robot.npz \
  --initial-robot-wrist private_inputs/screwdriver/baseline_robot.npz \
  --clip private_inputs/screwdriver/human_clip \
  --mesh private_inputs/screwdriver/screwdriver_object_m.obj \
  --output private_inputs/screwdriver/optimized_robot.npz \
  --steps 3000 --sdf-resolution 48
python -m rl.derive_candidate \
  --root private_inputs/screwdriver \
  --robot-model private_inputs/screwdriver/robot_model.npz
```

Retargeting defaults to a 500-step keypoint initializer, Adam learning rate 0.02, and 3000 joint-trajectory optimization steps. Laplacian, contact, penetration, and smoothness weights are 500, 20000, 100000, and 1. Fixed representative contacts, the distance kernel, vertex-distance SDF, convex-hull self-collision, and the regularizer are explicit implementation choices, not an exact rerun of the authors' code. The robot is Revo3 rather than the paper's Inspire, and the original ManipTrans is not included.

The portable exporter and HDF binding use the same FK and interpolation rules, but fresh exports, float32 FK, and metadata serialization may produce different file SHA256 hashes. Files generated through these public interfaces must not be described as byte-identical to the measured experiment. Numerical recomputation from the published trace is the result directly verifiable in this repository.

## 28 DOF and PPO Integration

Obtain the specified branch through authorized access at `private_inputs/screwdriver/baseline_repo`, pinned to `5c409994e4e04fd131aa46ad8441480da025c7fd`. The robot asset must match SHA256 `f8987c6c43e0f0a6fee02d13864b1bd59f0a976383ea8dce2da0dc46ba6c67ae`, and the original checkpoint hash must match `data/screwdriver/metadata.json`.

```bash
python -m rl.c2dex_reduced28_inputs \
  --root private_inputs/screwdriver --output-name reduced28_v2
```

This step recomputes 28 DOF IK for both wrist trajectories using the branch's base, FK, and joint limits. Do not directly reuse the ZIP's original 58-joint assembly sidecar. Both trajectories use the same 270-frame Makima/Slerp interpolation. The IK reachability report is diagnostic, not an additional physical success gate.

Install the external baseline package in an environment with Isaac Sim 5.1, Isaac Lab, torch CUDA, and rsl-rl 3.0.1, and select only one idle GPU. Native task construction still requires an authorized original bank path, but the adapter subsequently disables that bank; it is not used as training data for the new ZIP.

```bash
export REGRIND_ROOT="$PWD/private_inputs/screwdriver/baseline_repo"
export REGRIND_DATA_DIR="$REGRIND_ROOT/data"
export REGRIND_ARM_RESET_BANK_PATH="$REGRIND_DATA_DIR/precomputedik/augmented_arm_reset_bank_1024.npz"
export REGRIND_ARM_RESET_SOURCE_URDF_PATH="$REGRIND_ROOT/source/regrind/regrind/assets/tron2_axis180/assembly_bilateral_axis180_reduced28_physicsfix.urdf"
export PYTHONPATH="$REGRIND_ROOT/source/regrind:$PWD"
# Check that a GPU is idle, then set CUDA_VISIBLE_DEVICES to that GPU before running.
python -m rl.c2dex_reduced28_train \
  --inputs private_inputs/screwdriver/reduced28_v2/baseline \
  --output private_inputs/screwdriver/training_baseline \
  --checkpoint "$REGRIND_ROOT/checkpoint/model_5200.pt" \
  --updates 200 --num-envs 512 --headless
```

For retarget integration, change `--inputs` to `candidate` under the same root, use a separate exclusive output directory, and start from the same original checkpoint. The code disables DR, the external bank, perturbations, and augmentation while retaining the original reward/PPO architecture and training RSI. Integration code is provided, but training outcomes without successful pickup are excluded from the published comparison.

## Physical Evaluation Contract

New training requires the external baseline's actual physics evaluator. Neither training reward nor this repository's postprocessing is sufficient to declare task success. The published case uses the same branch's task control with a task-local evaluator adapter: start at frame zero, disable RSI/perturbations/DR, write the object only at initialization, prevent auto-reset, and record actual poses and object-filtered fingertip centroids. PPO actions are computed at reference frame t and advance to t+1.

The original complete task thresholds include lift 5 cm, hold height 3 cm for 0.5 s, position RMSE 3 cm and maximum 8 cm, rotation RMSE 15° and maximum 30°, plus correct initialization, completion of the full trajectory, valid physics, and no silent reset. The published case passes pickup/hold and position criteria but fails rotation. The full evaluator is not packaged as a one-command new physics rerun; the measured successful trace can be recomputed independently.
