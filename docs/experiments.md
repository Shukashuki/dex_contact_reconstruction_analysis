# 成功拿取任務的接觸重建實驗

本節回答兩個問題：重建是否改善螺絲刀 PPO，以及重建的接觸區域和成功拿取任務中的 sim 接觸點相距多遠。只分析成功拿取子集合；不列未成功拿取案例的比較表，也不從此子集合推算整體任務成功率。

## 螺絲刀 PPO 設定

輸入為前方桌面螺絲刀 ZIP。robot 統一使用指定 dex-rl 分支的 28 DOF physicsfix 設定，模擬完整重力，物體僅在初始化寫入，之後由物理演化。評估從第零幀開始，關閉 RSI、擾動與 domain randomization，確認沒有中途 reset。

PPO 從 `model_5200.pt` 開始微調，使用原分支的 actor、critic、控制方式與 reward，512 個環境、每次 24 步 rollout、200 次更新，共 2,457,600 transitions，seed 42。新微調只使用這份 ZIP，保留分支的訓練 RSI，但停用外部 demo 的 reset bank。原 checkpoint 曾使用另一份 demo，因此不能稱為從零開始的 ZIP-only 完整訓練。

本節的拿取條件是抬升至少 5 cm，且在初始高度上方 3 cm 持握至少 0.5 秒。這個條件和完整旋轉任務成功分開記錄。

| 成功拿取案例的物理量測 | 結果 |
| --- | ---: |
| 最大抬升 | 31.907 cm |
| 最長持握 | 7.333 秒 |
| 最終抬升 | 27.831 cm |
| 有接觸的控制影格 | 218 / 270 |
| 抬升超過 3 cm 且有接觸的影格 | 209 / 270 |
| 最大同時接觸手指數 | 4 |
| 最長連續三指接觸 | 1.133 秒 |
| 物體位置 tracking RMSE | 17.684 mm |

此案例達到拿取與持握子條件，位置 tracking 也在原評估門檻內；但旋轉姿態 RMSE 為 95.066°，因此完整旋轉任務仍未通過。不能將「拿起螺絲刀」寫成「完成旋轉螺絲刀」。

## 重建是否改善 PPO

目前結論為「尚未證實」，而不是改善。

上述成功拿取案例是原 ZIP baseline 經 PPO 微調的結果，並非 contact-preserving C2Dex retarget 的成功案例。依成功案例篩選後，沒有同 robot、同物體、同初始條件、同 checkpoint 與同預算，且重建開啟／關閉都成功拿取的配對。因此無法估計重建對 PPO、拿取表現或接觸一致性的改善，也不能把 PPO 微調後的成功歸因於重建。

後續若有配對成功案例，才比較有／無重建的持握接觸覆蓋、同幀同指距離與旋轉 tracking。只保留成功案例的結果是條件分析，不取代完整樣本上的方法有效性評估。

## 重建接觸點與成功持握接觸點

此 ZIP 沒有對應的相機參數與 silhouette reconstruction 輸入。本輪以 3D MANO keypoint fit、8 mm proximity 門檻與 opposite-normal filtering 建立接觸，再聚合成 canonical object space 的穩定區域。它是 C2Dex-style 接觸近似，不是原論文的單目 reconstruction benchmark。

MANO shape 未提供，採用零 betas 與骨長 least-squares scale。穩定重建標籤只涵蓋第 33 到 41 幀，共 9 / 270 幀，150 個 frame-expanded vertex constraints；成功持握階段有同步標籤的影格數為零。因此不能計算持握階段的同幀接觸預測準確率。

sim 接觸使用 PhysX 的物體過濾指尖 contact centroid，力大於 0.1 N 且座標有限才有效。它不是所有 raw narrowphase points。每個量測點使用當下實際 sim 物體姿態轉為物體座標；不能使用 reference 姿態替代。

目前可計算的是跨時間空間對照：對每個持握期 sim 接觸點，找較早重建的同指接觸區域中最近的點。它描述實際持握的位置與既有重建區域的差異，不是持握期接觸預測誤差。

| 跨時間同指區域對照 | 結果 |
| --- | ---: |
| 持握期有效指尖 centroid 數 | 534 |
| 有同指重建區域可比較的 centroid 數 | 372 |
| 平均最近區域距離 | 67.269 mm |
| P90 最近區域距離 | 115.191 mm |
| 距離在 20 mm 內 | 42 / 372，11.290% |

| 手指 | 持握 centroid 數 | 平均最近區域距離 |
| --- | ---: | ---: |
| 拇指 | 187 | 67.224 mm |
| 食指 | 115 | 97.958 mm |
| 中指 | 162 | 無重建區域，不計距離 |
| 無名指 | 69 | 17.116 mm |
| 小指 | 1 | 7.021 mm |

總平均按可比較的 372 個 centroid 加權，不是五指平均。中指的 162 個樣本不在距離分母中；小指只有一個樣本，不能以此判斷穩健性。沒有接觸或沒有同指區域時記為缺失，而非零距離。

![成功拿取與跨時間接觸區域對照](../results/screwdriver_success.png)

## 通過原專案物理門檻的 TACO 補充案例

兩個保留的 TACO 案例都由原報告確認 `ref_lifts`、`obj_lifted`、`held_aloft_frames > 0` 與 project physics gate 通過。這是浮動腕部 MuJoCo 回放，不是本節的 28 DOF Isaac PPO，也不能套用同一組成功門檻或直接排名。

| 成功拿取案例 | 抬升 | 持握影格 | 持握接觸 temporal IoU | 持握 symmetric contact distance |
| --- | ---: | ---: | ---: | ---: |
| TACO 20231013_304 左手 | 3.564 cm | 2 | 1.000 | 119.055 mm |
| TACO 20231104_156 右手 | 23.855 cm | 77 | 0.987 | 47.813 mm |

TACO 距離使用同幀、物體座標系的雙向 point-set mean，沒有手指 correspondence；sim 點來自 narrowphase，且在每控制幀以 1 mm 去重。這和螺絲刀的跨時間、同指、單向 centroid-to-region 距離不同，不能當成同一指標比較大小。第一例的 ray extraction stride 為 5，第二例為全幀。

這兩例顯示，物理上能拿起物體，不代表接觸區域完全保留。它們也不構成重建開啟／關閉的成對改善實驗。完整量測見 [TACO press](../data/taco_success/taco_press.json) 與 [TACO stir fry](../data/taco_success/taco_stir_fry.json)。

## 證據與限制

螺絲刀分析可用附帶的衍生 trace 重新計算，輸入 SHA256 會在分析前驗證。只有一個 seed 和有限 PPO 微調預算；重建存在相機、MANO shape、穩定標籤時間範圍的缺口。原始完整旋轉成功判定仍為 false。這些限制不能由接觸距離或成功拿取的事後篩選補足。
