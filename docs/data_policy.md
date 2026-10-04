# 復現資料與資產範圍

本 repo 提供使用者指定實驗的成功拿取衍生量測，足以重算公開分析。未提供完整原始 dataset，也沒有授權第三方資產的重新散布。

## 已提供資料

- `data/screwdriver/successful_ppo/assembly_trace.npz`：270 幀 sim 物體姿態、reference 物體姿態、指尖物體接觸力、contact centroid 與 fps；僅數值陣列，不含 pickle。
- `data/screwdriver/predictions.npz`：150 個穩定接觸約束的 frame、finger 與 canonical object point。
- `data/screwdriver/metadata.json`：原 ZIP、robot、checkpoint 與衍生資料的 SHA256、量測定義。
- `data/screwdriver/training.json`：成功 baseline 的 PPO 預算與 checkpoint lineage，不含機器路徑。
- `data/taco_success/`：兩個通過各自原專案物理門檻的 TACO 衍生報告；不包含原始影像或 MANO 權重。
- `data/revo3_contract.json`：關節名稱、limits、keypoint 對應與腕部座標旋轉；不含 collision mesh。

## 未提供資產

MANO pickle／NPZ 權重、原始螺絲刀 ZIP、完整 TACO、robot STL／URDF／USD、PPO checkpoint 與外部 baseline source 都須自行依各自授權取得。`.gitignore` 排除本地模型與資產，但使用者仍須自行確認取得與使用權限。

螺絲刀原始 trace 與本 repo 數值子集合的 SHA256 不同是預期的：公開版只保留分析所需數值欄位，移除其他 trace 欄位與本地路徑。metadata 分別記錄兩者；公開版會重新驗證並重算距離。

本 repo 只發佈成功拿取子集合的實驗資料。它不是全樣本 benchmark，不能用保留案例數量估計原實驗的整體成功率。
