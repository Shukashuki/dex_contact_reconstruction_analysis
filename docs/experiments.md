# Contact Reconstruction Experiments on Successful Pickup Tasks

This section asks whether reconstruction improves screwdriver PPO and how far reconstructed contact regions are from simulated contacts in successful pickup tasks. Only the successful-pickup subset is analyzed. Unsuccessful pickups are excluded from comparison tables, and overall task success rates are not inferred from this subset.

## Screwdriver PPO Setup

The input is the front-table screwdriver ZIP. The robot uses the specified dex-rl branch's 28 DOF physicsfix configuration, with full gravity. The object state is written only at initialization and evolves through physics afterward. Evaluation starts at frame zero with RSI, perturbations, and domain randomization disabled, and checks that no intermediate resets occur.

PPO fine-tuning starts from `model_5200.pt`, retaining the branch's actor, critic, control scheme, and reward: 512 environments, 24 rollout steps per update, 200 updates, 2,457,600 transitions, and seed 42. New fine-tuning uses only this ZIP and retains training RSI while disabling the external-demo reset bank. The original checkpoint used another demo, so this is not complete ZIP-only training from scratch.

Pickup requires a lift of at least 5 cm and a hold at least 3 cm above the initial height for at least 0.5 seconds. This is recorded separately from success on the complete rotation task.

| Physical measurement for the successful pickup | Result |
| --- | ---: |
| Maximum lift | 31.907 cm |
| Longest hold | 7.333 seconds |
| Final lift | 27.831 cm |
| Control frames with contact | 218 / 270 |
| Frames with lift above 3 cm and contact | 209 / 270 |
| Maximum simultaneous contacting fingers | 4 |
| Longest continuous three-finger contact | 1.133 seconds |
| Object position tracking RMSE | 17.684 mm |

This case meets pickup and hold subcriteria, and position tracking is within the original evaluation threshold. Rotation RMSE is 95.066°, however, so the complete rotation task still fails. Picking up the screwdriver must not be described as completing screwdriver rotation.

## Does Reconstruction Improve PPO

The current conclusion is “not established,” not improvement.

The successful pickup above comes from PPO fine-tuning of the original ZIP baseline, not a successful contact-preserving C2Dex retarget. After success filtering, there is no matched reconstruction-enabled/disabled pair with the same robot, object, initial conditions, checkpoint, and budget in which both runs achieve pickup. Benefits for PPO, pickup performance, or contact consistency therefore cannot be estimated, and PPO success cannot be attributed to reconstruction.

If matched successful pairs become available, comparisons can cover held-phase contact coverage, same-frame same-finger distances, and rotation tracking with and without reconstruction. Success-conditioned analysis does not replace method evaluation on the complete sample.

## Reconstructed Contacts Versus Successful Held Contacts

The ZIP lacks corresponding camera parameters and silhouette reconstruction inputs. Contacts are constructed using 3D MANO keypoint fitting, an 8 mm proximity threshold, and opposite-normal filtering, then aggregated into stable regions in canonical object space. This is a C2Dex-style approximation, not the paper's monocular reconstruction benchmark.

MANO shape is unavailable, so zero betas and a bone-length least-squares scale are used. Stable reconstruction labels cover frames 33 through 41: 9 / 270 frames and 150 frame-expanded vertex constraints. There are zero synchronized-label frames during successful holding, so same-frame held-phase contact prediction accuracy cannot be computed.

Simulated contacts are PhysX object-filtered fingertip contact centroids, valid only with force above 0.1 N and finite coordinates. These are not all raw narrowphase points. Each measurement is transformed into object coordinates using the actual simulated object pose at that time, not the reference pose.

The available metric is a cross-time spatial comparison: for each held-phase simulated contact, find the nearest point in the earlier reconstructed region for the same finger. This describes differences between actual holding locations and existing reconstructed regions, not held-phase prediction error.

| Cross-time same-finger region comparison | Result |
| --- | ---: |
| Valid held-phase fingertip centroids | 534 |
| Centroids with a same-finger reconstructed region | 372 |
| Mean nearest-region distance | 67.269 mm |
| P90 nearest-region distance | 115.191 mm |
| Distance within 20 mm | 42 / 372, 11.290% |

| Finger | Held-phase centroids | Mean nearest-region distance |
| --- | ---: | ---: |
| Thumb | 187 | 67.224 mm |
| Index | 115 | 97.958 mm |
| Middle | 162 | No reconstructed region; excluded from distance |
| Ring | 69 | 17.116 mm |
| Little | 1 | 7.021 mm |

The overall mean is weighted by the 372 comparable centroids, not averaged over five fingers. The 162 middle-finger samples are excluded from the distance denominator. The little finger has only one sample, insufficient to establish robustness. Missing contacts or same-finger regions are recorded as missing, not zero distance.

![Successful pickup and cross-time contact-region comparison](../results/screwdriver_success.png)

## Additional TACO Cases Passing Their Original Physics Gates

The original reports confirm `ref_lifts`, `obj_lifted`, `held_aloft_frames > 0`, and a passing project physics gate for both retained TACO cases. These are floating-wrist MuJoCo replays, not the 28 DOF Isaac PPO setup above. They do not share the same success thresholds and cannot be ranked directly.

| Successful pickup case | Lift | Held frames | Held-contact temporal IoU | Held symmetric contact distance |
| --- | ---: | ---: | ---: | ---: |
| TACO 20231013_304 left hand | 3.564 cm | 2 | 1.000 | 119.055 mm |
| TACO 20231104_156 right hand | 23.855 cm | 77 | 0.987 | 47.813 mm |

TACO distances are same-frame, bidirectional point-set means in object coordinates without finger correspondence. Simulated points come from narrowphase contacts and are deduplicated at 1 mm within each control frame. This differs from the screwdriver's cross-time, same-finger, unidirectional centroid-to-region distance; the numbers must not be ranked as the same metric. Ray extraction uses stride 5 in the first case and all frames in the second.

These cases show that physical pickup does not imply full contact-region preservation. They are not a paired reconstruction-enabled/disabled improvement experiment. Full measurements are in [TACO press](../data/taco_success/taco_press.json) and [TACO stir fry](../data/taco_success/taco_stir_fry.json).

## Evidence and Limitations

The screwdriver analysis can be recomputed from the included derivative trace, with input SHA256 hashes verified before analysis. There is only one seed and a bounded PPO fine-tuning budget. Reconstruction has gaps in camera inputs, MANO shape, and stable-label temporal coverage. The complete rotation-task success verdict remains false. Contact distances and post hoc success filtering cannot resolve these limitations.
