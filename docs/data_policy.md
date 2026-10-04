# Reproduction Data and Asset Scope

This repository provides derivative measurements from successful pickups in the user-selected experiments, sufficient to recompute the published analysis. It does not provide the complete raw dataset or authorize redistribution of third-party assets.

## Included Data

- `data/screwdriver/successful_ppo/assembly_trace.npz`: 270 frames of simulated and reference object poses, fingertip-to-object contact forces, contact centroids, and fps; numeric arrays only, without pickle.
- `data/screwdriver/predictions.npz`: frame, finger, and canonical object point for 150 stable contact constraints.
- `data/screwdriver/metadata.json`: SHA256 hashes of the original ZIP, robot, checkpoint, and derivatives, plus measurement definitions.
- `data/screwdriver/training.json`: PPO budget and checkpoint lineage for the successful baseline, without machine-specific paths.
- `data/taco_success/`: derivative reports for two TACO cases passing their original project physics gates; no raw images or MANO weights.
- `data/revo3_contract.json`: joint names, limits, keypoint mappings, and wrist-frame rotation; no collision meshes.

## Assets Not Included

MANO pickle/NPZ weights, the original screwdriver ZIP, the complete TACO dataset, robot STL/URDF/USD assets, PPO checkpoints, and external baseline source must be obtained separately under their respective licenses. Although `.gitignore` excludes local models and assets, users must still verify their access and usage rights.

Different SHA256 hashes for the original screwdriver trace and this repository's numeric subset are expected. The public version retains only numeric fields needed for analysis and removes other trace fields and local paths. Metadata records both hashes; the published inputs are verified again before recomputing distances.

Only the successful-pickup subset is published. This is not a full-sample benchmark, and retained case counts cannot estimate overall success rates in the original experiments.
