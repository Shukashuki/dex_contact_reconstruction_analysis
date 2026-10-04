# C2Dex Contact Reconstruction and Successful Pickup Analysis

This repository contains an independent implementation of C2Dex contact reconstruction and retargeting, plus comparisons between reconstructed contacts and simulated contacts in successful pickup tasks. Only cases meeting their respective pickup criteria are included; unsuccessful pickups are excluded from contact comparisons.

After PPO fine-tuning, the screwdriver baseline lifts the object by 31.9 cm and holds it for 7.33 seconds. Its held-phase contacts have a mean nearest distance of 67.3 mm to earlier reconstructed regions, matched by finger in object coordinates. There are no synchronized reconstruction labels during holding or matched pairs where both reconstruction-enabled and reconstruction-disabled runs achieve pickup. These results do not establish that reconstruction improves PPO, and this distance is not held-phase prediction accuracy.

## Experiments and Code

- [Experiments](docs/experiments.md): screwdriver PPO, contact distances in successful pickups, and evidence limitations.
- [Reproduction](docs/reproduction.md): analysis, MANO loading, reconstruction, retargeting, and 28 DOF PPO integration.
- [Data and Asset Scope](docs/data_policy.md): included derivatives and assets that must be obtained separately.
- [Successful screwdriver results](results/screwdriver_success.json) and [contact data](results/screwdriver_success.contacts.npz).

![Lift, hold, and contact regions during successful pickup](results/screwdriver_success.png)

## Rerun the Successful Case Analysis

Recompute contact distances from the included derivative trace without a GPU, Isaac Sim, MANO models, or the original ZIP.

```bash
python -m pip install -e '.[test]'
python -m analysis.successful_screwdriver
python -m pytest -q
```

Reconstruction and retraining require legally obtained MANO, robot, and trajectory assets; see the reproduction guide. Providing code does not mean the complete paper pipeline has been reproduced. This screwdriver experiment uses 3D keypoint fitting and proximity-based contact approximation. PPO is bounded-budget fine-tuning of the specified dex-rl checkpoint, not the original C2Dex ManipTrans training.

## Sources

The framework follows the [C2Dex paper](https://arxiv.org/abs/2608.07045v2). The PPO baseline uses the [specified dex-rl branch](https://github.com/clearlab-sustech/dex-rl/tree/hand-and-tron2-RL-regrind-reward), pinned to commit `5c409994e4e04fd131aa46ad8441480da025c7fd`. Experimental numbers come from the included measurements, not the paper benchmarks.
