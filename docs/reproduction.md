# 接觸重建與成功案例分析的復現方法

分析可以直接重跑；完整重建與新 PPO 訓練則需要本地取得的原始資料、MANO、MuJoCo robot、指定 dex-rl 分支與 Isaac 環境。這些資產未打包在 repo 中。

## 重跑公開成功案例

```bash
python -m pip install -e '.[test]'
python -m analysis.successful_screwdriver
python -m pytest -q
```

入口會檢查拿取子條件與輸入 SHA256，然後重算物體座標中的接觸、同步標籤覆蓋、持握區域距離和圖。沒有同指預測區域時，距離為 null，不是零。輸出為 `results/screwdriver_success.json`、PNG 與 contact NPZ。

## 本地 MANO

```bash
python -m pip install -e '.[reconstruction,robot,test]'
python -m perception.export_mano \
  --root private_inputs/mano/models --side right \
  --output perception/hand_artifacts/mano_right.npz
```

pickle 只能從可信、合法來源載入。可用 `MANO_ROOT` 指定 pickle 資料夾，`MANO_NPZ_ROOT` 指定匯出的 NPZ 資料夾。不要將權重提交到 repo。

## 螺絲刀的重建輸入

準備原 ZIP、對應 canonical meter object OBJ 與桌面 metadata。原 ZIP SHA256 固定為 `4e2735400b5e44a28ac6f05bda926693ed2944c6dec787b1f7c8b0c32f8f92cb`；其他檔案不可冒充該實驗輸入。

```bash
python -m perception.screwdriver_fit --prepare \
  --root private_inputs/screwdriver \
  --archive private_inputs/front_table_screwdriver.zip \
  --mesh private_inputs/screwdriver_object_m.obj \
  --scene data/screwdriver/scene_from_object.json
python -m perception.screwdriver_fit --root private_inputs/screwdriver
```

prepare 只抽取 HDF 與 sidecar，保留 object 與 MANO reference，不抽取影片。fit 使用零 MANO betas、骨長 least-squares scale、3D keypoint fitting、8 mm proximity 與 opposite-normal score 大於 0.5 的條件，再用穩定片段／dominant cluster 聚合接觸。原紀錄 scale 1.65208、fit mean 2.839 mm、median 1.705 mm、P90 8.049 mm；這是 keypoint fitting，不是 sim 接觸誤差。

`perception/c2dex_reconstruction.py` 另提供有相機輸入時的 ray extraction，`perception/c2dex_optimize.py` 提供 MANO trajectory optimization。本輪螺絲刀成功案例的標籤只來自上述 3D fit 和 proximity 路徑，不能因 repo 提供完整模組就宣稱該案例執行了 silhouette-ray reconstruction 或 reconstruction trajectory optimization。

## 接觸保留 retarget

先用本地合法的固定基座 Revo3 MuJoCo XML 匯出模型。`data/revo3_contract.json` 明確指定關節、keypoint 與 wrist-to-handroot rotation，不推測座標轉換。

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

retarget 預設使用 500-step keypoint initializer、Adam learning rate 0.02 與 3000 次 joint-trajectory optimization。Laplacian、contact、penetration、smoothness 的權重為 500、20000、100000、1。固定代表接觸點、distance kernel、vertex-distance SDF、convex hull self-collision 與 regularizer 都是本實作的明確選擇，不是精確原作者 code rerun。robot 使用 Revo3 而非論文的 Inspire；沒有原始 ManipTrans。

portable exporter／HDF binding 使用相同 FK 與插值規則，但 fresh export、float32 FK 與 metadata 序列化可能產生新的檔案 SHA256。不能宣稱從這些公開介面產生的檔案與已量測實驗逐 byte 相同；公開 trace 的數值重算才是本 repo 可直接驗證的結果。

## 28 DOF 與 PPO 接入

將指定分支以合法存取方式取得到 `private_inputs/screwdriver/baseline_repo`，固定到 `5c409994e4e04fd131aa46ad8441480da025c7fd`。資產需符合 robot SHA256 `f8987c6c43e0f0a6fee02d13864b1bd59f0a976383ea8dce2da0dc46ba6c67ae`，原 checkpoint SHA256 需符合 `data/screwdriver/metadata.json`。

```bash
python -m rl.c2dex_reduced28_inputs \
  --root private_inputs/screwdriver --output-name reduced28_v2
```

這一步使用該分支的 base、FK 與 joint limits，為兩組 wrist 軌跡重新求解 28 DOF IK。不可直接使用 ZIP 中原 58-joint assembly 的 sidecar。兩組採用相同 270 幀 Makima／Slerp 插值；IK reachability report 是診斷，不是額外物理成功門檻。

在具有 Isaac Sim 5.1、Isaac Lab、torch CUDA、rsl-rl 3.0.1 的環境中安裝外部 baseline package，並確認只指定一張空閒 GPU。native task 建構時仍需合法的原 bank 路徑，但 adapter 隨後會將它停用，不作為新 ZIP 訓練資料。

```bash
export REGRIND_ROOT="$PWD/private_inputs/screwdriver/baseline_repo"
export REGRIND_DATA_DIR="$REGRIND_ROOT/data"
export REGRIND_ARM_RESET_BANK_PATH="$REGRIND_DATA_DIR/precomputedik/augmented_arm_reset_bank_1024.npz"
export REGRIND_ARM_RESET_SOURCE_URDF_PATH="$REGRIND_ROOT/source/regrind/regrind/assets/tron2_axis180/assembly_bilateral_axis180_reduced28_physicsfix.urdf"
export PYTHONPATH="$REGRIND_ROOT/source/regrind:$PWD"
# 在 shell 外先確認 GPU 空閒，再設定 CUDA_VISIBLE_DEVICES 為該張 GPU。
python -m rl.c2dex_reduced28_train \
  --inputs private_inputs/screwdriver/reduced28_v2/baseline \
  --output private_inputs/screwdriver/training_baseline \
  --checkpoint "$REGRIND_ROOT/checkpoint/model_5200.pt" \
  --updates 200 --num-envs 512 --headless
```

retarget 接入可將 `--inputs` 換為同 root 下的 `candidate`，使用另一個 exclusive output 並從相同原 checkpoint 起跑。程式會停用 DR、外部 bank、擾動與 augmentation，保留原 reward／PPO 架構及訓練 RSI。只提供接入程式，不將未成功拿取的訓練結果列入公開比較。

## 物理評估契約

重新訓練後還需外部 baseline 的實際物理 evaluator，不能只用 training reward 或本 repo 的 postprocessing 宣告任務成功。已發佈案例使用同分支的任務控制，加上 task-local evaluator adapter：從零幀開始、無 RSI／擾動／DR、物體只在初始化寫入、阻止 auto-reset，並記錄实际姿態和 object-filtered fingertip centroids。PPO action 在 reference frame t 計算，推進到 t+1。

原完整任務門檻包含 lift 5 cm、hold height 3 cm／0.5 s、position RMSE 3 cm／max 8 cm、rotation RMSE 15°／max 30°，以及初始姿態正確、全軌跡結束、物理有效、無 silent reset。公開成功案例通過拿取／持握與位置條件，但旋轉失敗；完整 evaluator 不打包成可一鍵新 physics rerun，已量測的成功 trace 則可獨立重算。
