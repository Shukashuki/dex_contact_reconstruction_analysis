"""批次 MANO IK(torch/GPU)—— 取代 `contact_pipeline` 的逐幀 scipy 版。

CPU 版 2.58 s/frame,8 支 TACO 雙手花 59 分鐘;TACO 全集 2317 序列不可行。
更重要的是逐幀 `least_squares` 是**貪婪**的:每幀只用前一幀 warm start,
不保證整段一致。`retarget/collision_clamp.py` 的全軌跡 POCS 與 C2Dex 的
全序列聯合最佳化都指出同一件事 —— 時序耦合要放進同一個問題裡解。

torch 版的三個差異:

* **全序列同時解**:所有幀是一個張量,時序項是真正的耦合而非 warm start;
* **autograd 解析 Jacobian**:CPU 版用數值差分,每次迭代 61 次 forward;
* **GPU 批次**:MANO 是 778 頂點的線性代數,批次維度免費。

沿用 CPU 版被實測支持的設定(見 `perception/contact_pipeline.py`):
關節界(無界 IK 殘差 20.99 mm → 有界 1.70 mm)與時序一致性。
**不**採用 topo_retarget 的 interaction graph —— 該檔檔頭的控制實驗顯示
物體節點是輕微退步。

    python -m perception.mano_torch --clip taco_20231020_254 --side right
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 指尖取自網格頂點(與 perception.mano_layer.FINGERTIP_VERTICES 同一組)。
# ⚠ 只擬合 16 個 MANO 關節會**欠定**:51 個自由度對 48 個殘差,最佳化在
# 零空間游走(實測停滯於 16-35 mm,而同一目標的精確解存在 —— 把 scipy 的
# 解代入本模型殘差是 0.000 mm)。補上 5 個指尖 → 63 殘差,問題變成近乎
# 恰定,與 CPU 版一致。
FINGERTIP_VERTICES = (745, 317, 444, 556, 673)   # thumb, index, middle, ring, pinky
FINGER_POSE_LIMIT = 1.8
TRANS_LIMIT = 1.5
BETA_LIMIT = 3.0


def axis_angle_to_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """(...,3) 軸角 → (...,3,3)。Rodrigues,**在 θ=0 處可微**。

    ⚠ `vectors.norm()` 在 v=0 不可微,autograd 在該點給出錯誤梯度。實測:
    以 pose=0 為初值時 autograd Jacobian 與有限差分相對誤差 **13%**,
    且部分行範數為 0(autograd 認為那些參數不影響殘差)。TRF 從壞掉的
    二次模型出發就再也收斂不了 —— 這正是先前停滯在 30 mm 的根因。
    scipy 用有限差分,永遠踩不到這個點,所以同一問題它解得出來。

    修法:在平方和裡加 eps 再開根號,使函數處處平滑。
    """
    theta = torch.sqrt((vectors ** 2).sum(-1, keepdim=True) + 1e-16)
    axis = vectors / theta
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero],
                    dim=-1).reshape(*vectors.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=vectors.dtype, device=vectors.device)
    theta = theta.unsqueeze(-1)
    return eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def load_mano_npz(side: str) -> dict:
    """讀已 dump 的 MANO 陣列。torch 環境沒有 scipy(chumpy pickle 需要),
    所以模型由主 venv 一次匯出成 npz,兩邊共用同一份數值。"""
    path = Path(os.environ.get("MANO_NPZ_ROOT", str(PROJECT_ROOT / "perception" / "hand_artifacts"))) / f"mano_{side}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在 —— 先在主 venv 執行 mano dump(見檔頭)")
    data = np.load(path)
    return type("Mano", (), {k: data[k] for k in data.files})


class ManoTorch(torch.nn.Module):
    """MANO forward(LBS),批次維度在最前面。與 `perception.mano_layer` 同模型。"""

    def __init__(self, model, device: torch.device, dtype=torch.float32):
        super().__init__()
        def t(x):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        self.v_template = t(model.v_template)
        self.shapedirs = t(model.shapedirs)
        self.posedirs = t(model.posedirs)
        self.J_regressor = t(model.J_regressor)
        self.weights = t(model.weights)
        self.hands_mean = t(model.hands_mean)
        self.parents = [int(p) for p in np.asarray(model.parents)]
        self.faces = np.asarray(model.faces)

    def forward(self, pose: torch.Tensor, betas: torch.Tensor,
                trans: torch.Tensor, *, flat_hand_mean: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """pose (B,48) 軸角, betas (10,), trans (B,3) → (joints (B,16,3), verts (B,778,3)).

        ⚠ `flat_hand_mean=False`(預設)會把 `hands_mean` 加進手指 DOF,與
        `perception.mano_layer.forward` 一致。漏掉會讓整隻手蜷曲 ——
        實測 max|diff| 80.8 mm(該檔檔頭亦警告 "curls the whole hand")。
        """
        B = pose.shape[0]
        if not flat_hand_mean:
            pose = torch.cat([pose[:, :3], pose[:, 3:] + self.hands_mean], dim=1)
        shaped = self.v_template + torch.einsum(
            "vki,i->vk", self.shapedirs, betas)               # (778,3)
        joints0 = torch.einsum("jv,vk->jk", self.J_regressor, shaped)  # (16,3)

        rot = axis_angle_to_matrix(pose.reshape(B, 16, 3))     # (B,16,3,3)
        eye = torch.eye(3, dtype=rot.dtype, device=rot.device)
        feature = (rot[:, 1:] - eye).reshape(B, -1)            # (B,135)
        posed = shaped + torch.einsum("vkp,bp->bvk", self.posedirs, feature)

        # 沿運動樹累積全域變換
        global_rot = [rot[:, 0]]
        global_pos = [joints0[0].expand(B, 3)]
        for j in range(1, 16):
            parent = self.parents[j]
            global_rot.append(global_rot[parent] @ rot[:, j])
            offset = (joints0[j] - joints0[parent]).expand(B, 3)
            global_pos.append(
                global_pos[parent]
                + torch.einsum("bik,bk->bi", global_rot[parent], offset))
        R = torch.stack(global_rot, dim=1)                     # (B,16,3,3)
        P = torch.stack(global_pos, dim=1)                     # (B,16,3)

        # LBS:每個頂點在 rest 位置相對各關節的偏移,以蒙皮權重混合
        rest = joints0.unsqueeze(0)                            # (1,16,3)
        offsets = posed.unsqueeze(2) - rest.unsqueeze(1)       # (B,778,16,3)
        transformed = torch.einsum("bjik,bvjk->bvji", R, offsets) + P.unsqueeze(1)
        verts = torch.einsum("vj,bvji->bvi", self.weights, transformed)
        return P + trans.unsqueeze(1), verts + trans.unsqueeze(1)


def fit_sequence(model, target_joints: np.ndarray, *, device: str = "cuda",
                 iters: int = 600, temporal: float = 0.02,
                 lr: float = 0.05, verbose: bool = False) -> dict:
    """全序列聯合擬合。target_joints (T,16,3) 為 MANO 關節順序的觀測值。"""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    mano = ManoTorch(model, dev)
    target = torch.as_tensor(target_joints, dtype=torch.float32, device=dev)
    T = target.shape[0]

    # ⚠ 初值:MANO 的 joint 0 在 rest_joints[0] 而非原點(center_idx=None),
    # 所以 trans 必須扣掉該偏移,否則整隻手一開始就差一個模板位移。
    with torch.no_grad():
        rest0 = mano(torch.zeros(1, 48, device=dev),
                     torch.zeros(10, device=dev),
                     torch.zeros(1, 3, device=dev))[0][0, 0]
    pose = torch.zeros(T, 48, device=dev, requires_grad=True)
    trans = (target[:, 0] - rest0).clone().requires_grad_(True)
    betas = torch.zeros(10, device=dev, requires_grad=True)

    # Adam 在此問題上 step 200 就停滯於 36 mm —— 這是最小平方問題,
    # 一階法不利用其結構。LBFGS(擬牛頓)與 CPU 版的 trust-region 同族,
    # 實測把殘差帶回 mm 級。關節界改用軟懲罰,因為 LBFGS 不接受硬 clamp
    # 在 closure 之外的修改。
    optim = torch.optim.LBFGS([pose, trans, betas], lr=1.0,
                              max_iter=iters, history_size=50,
                              tolerance_grad=1e-9, tolerance_change=1e-12,
                              line_search_fn="strong_wolfe")

    state = {"step": 0}

    def closure():
        optim.zero_grad()
        joints, _ = mano(pose, betas, trans)
        fit = ((joints - target) ** 2).sum(-1).mean()
        smooth = ((pose[1:, 3:] - pose[:-1, 3:]) ** 2).mean() if T > 1 else 0.0
        # 軟界:超出解剖範圍才受罰(CPU 版實測硬界 20.99 → 1.70 mm)
        excess = (pose[:, 3:].abs() - FINGER_POSE_LIMIT).clamp(min=0)
        bound = (excess ** 2).mean()
        loss = fit + temporal * smooth + 10.0 * bound
        loss.backward()
        if verbose and state["step"] % 50 == 0:
            print(f"  iter {state['step']:4d} fit {fit.sqrt().item()*1000:6.2f} mm",
                  flush=True)
        state["step"] += 1
        return loss

    optim.step(closure)

    with torch.no_grad():
        joints, verts = mano(pose, betas, trans)
        error = (joints - target).norm(dim=-1).mean(dim=1) * 1000
    return {"pose": pose.detach().cpu().numpy(),
            "betas": betas.detach().cpu().numpy(),
            "trans": trans.detach().cpu().numpy(),
            "vertices": verts.cpu().numpy(),
            "fit_mm": error.cpu().numpy(), "device": str(dev)}


def fit_sequence_lm(model, target_joints: np.ndarray, *, device: str = "cuda",
                    iters: int = 60, lam0: float = 1e-3,
                    verbose: bool = False) -> dict:
    """批次 Levenberg–Marquardt —— 每幀各自解正規方程,GPU 上平行。

    為什麼不是 Adam / LBFGS(兩者都試過並記錄):

        Adam(全序列)      31.15 mm  —— 一階法,step 200 即停滯
        LBFGS(全序列)     16.59 mm  —— 1500 迭代僅比 400 好 0.7 mm
        scipy trf(逐幀)    1.70 mm  —— 但 2580 ms/frame

    差異不在 betas 自由度(逐幀 betas 實測 23.77 mm,**比共用的 14.54 更差**,
    假設已否定),而在**問題結構**:`trf` 是最小平方信賴域法且每幀是一個
    獨立良態小問題;LBFGS 把 190 幀綁成一個 9703 維問題共用一組歷史。

    本函式保留「每幀獨立」的良態性,同時用批次張量拿到 GPU 平行:
    J (B,48,51) 與 r (B,48) 逐幀組裝,(JᵀJ+λI)δ = −Jᵀr 以 batched solve 求解。
    betas 共用(同一隻手不會逐幀變形),在外層交替更新。
    """
    from torch.func import jacrev, vmap

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    mano = ManoTorch(model, dev, dtype=torch.float64)
    target = torch.as_tensor(target_joints, dtype=torch.float64, device=dev)
    T = target.shape[0]

    with torch.no_grad():
        rest0 = mano(torch.zeros(1, 48, dtype=torch.float64, device=dev),
                     torch.zeros(10, dtype=torch.float64, device=dev),
                     torch.zeros(1, 3, dtype=torch.float64, device=dev))[0][0, 0]
    x = torch.zeros(T, 51, dtype=torch.float64, device=dev)
    x[:, 48:] = target[:, 0] - rest0
    if target.shape[1] != 21:
        raise ValueError(
            f"target 需為 (T,21,3):16 MANO 關節 + 5 指尖,得到 {tuple(target.shape)}")
    betas = torch.zeros(10, dtype=torch.float64, device=dev)

    def frame_residual(xi, tgt, b):
        joints, verts = mano(xi[:48].unsqueeze(0), b, xi[48:].unsqueeze(0))
        tips = verts[0, list(FINGERTIP_VERTICES)]
        full = torch.cat([joints[0], tips], dim=0)      # (21,3)
        return (full - tgt).reshape(-1)

    jac = vmap(jacrev(frame_residual), in_dims=(0, 0, None))
    res_fn = vmap(frame_residual, in_dims=(0, 0, None))
    lam = torch.full((T, 1, 1), lam0, dtype=torch.float64, device=dev)
    eye = torch.eye(51, dtype=torch.float64, device=dev)

    for step in range(iters):
        r = res_fn(x, target, betas)                      # (T,48)
        J = jac(x, target, betas)                         # (T,48,51)
        JT = J.transpose(1, 2)
        H = JT @ J + lam * eye
        g = -(JT @ r.unsqueeze(-1))
        delta = torch.linalg.solve(H, g).squeeze(-1)
        x_new = x + delta
        r_new = res_fn(x_new, target, betas)
        better = (r_new ** 2).sum(1) < (r ** 2).sum(1)    # 逐幀各自接受/拒絕
        x = torch.where(better.unsqueeze(1), x_new, x)
        lam = torch.where(better.reshape(-1, 1, 1), lam * 0.5, lam * 3.0)
        lam = lam.clamp(1e-10, 1e6)
        # 軟界:超出解剖範圍的手指姿態拉回(對應 CPU 版的硬界)
        x[:, 3:48] = x[:, 3:48].clamp(-FINGER_POSE_LIMIT, FINGER_POSE_LIMIT)
        if verbose and step % 10 == 0:
            error = (r.reshape(T, -1, 3).norm(dim=-1).mean()) * 1000
            print(f"  lm {step:3d}  fit {error.item():6.2f} mm", flush=True)

    with torch.no_grad():
        joints, verts = mano(x[:, :48], betas, x[:, 48:])
        tips = verts[:, list(FINGERTIP_VERTICES)]
        full = torch.cat([joints, tips], dim=1)          # (T,21,3),與殘差同序
        error = (full - target).norm(dim=-1).mean(dim=1) * 1000
    return {"pose": x[:, :48].cpu().numpy(), "trans": x[:, 48:].cpu().numpy(),
            "betas": betas.cpu().numpy(), "vertices": verts.cpu().numpy(),
            "fit_mm": error.cpu().numpy(), "device": str(dev)}


def fit_sequence_trf(model, target_joints: np.ndarray, *, device: str = "cuda",
                     max_iter: int = 120, verbose: bool = False) -> dict:
    """用 `perception.trf_torch` 的信賴域法批次擬合 —— 取代簡化 LM。

    簡化 LM(`fit_sequence_lm`)實測停滯於 30-35 mm,而精確解存在。差別在
    信賴域管理:gain ratio、變數縮放、以 Δ 為主而非直接調 λ。三者見
    `perception/trf_torch.py` 檔頭。
    """
    from torch.func import jacrev, vmap

    from perception.trf_torch import batched_trf

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    # 稀疏前向:Jacobian 只需要 16 關節 + 5 指尖,不必走 778 頂點的 LBS
    mano = ManoSparse(model, FINGERTIP_VERTICES, dev)
    target = torch.as_tensor(target_joints, dtype=torch.float64, device=dev)
    T = target.shape[0]
    if target.shape[1] != 21:
        raise ValueError(f"target 需為 (T,21,3),得到 {tuple(target.shape)}")

    with torch.no_grad():
        rest0 = mano(torch.zeros(1, 48, dtype=torch.float64, device=dev),
                     torch.zeros(1, 3, dtype=torch.float64, device=dev))[0][0, 0]

    def frame_residual(xi, tgt):
        joints, tips = mano(xi[:48].unsqueeze(0), xi[48:].unsqueeze(0))
        full = torch.cat([joints[0], tips[0]], dim=0)
        return (full - tgt).reshape(-1)

    res_v = vmap(frame_residual, in_dims=(0, 0))
    jac_v = vmap(jacrev(frame_residual), in_dims=(0, 0))

    x0 = torch.zeros(T, 51, dtype=torch.float64, device=dev)
    x0[:, 48:] = target[:, 0] - rest0
    lower = torch.cat([torch.full((3,), -np.pi), torch.full((45,), -FINGER_POSE_LIMIT),
                       torch.full((3,), -TRANS_LIMIT)]).to(dev).expand(T, 51)
    upper = -lower

    out = batched_trf(lambda x: res_v(x, target), lambda x: jac_v(x, target),
                      x0, lower=lower, upper=upper, max_iter=max_iter,
                      verbose=verbose)
    x = out["x"]
    with torch.no_grad():
        joints, tips = mano(x[:, :48], x[:, 48:])
        full = torch.cat([joints, tips], dim=1)
        error = (full - target).norm(dim=-1).mean(dim=1) * 1000
        # 完整網格只在最後算一次(接觸估計需要全部 778 頂點)
        dense = ManoTorch(model, dev, dtype=torch.float64)
        _, verts = dense(x[:, :48], torch.zeros(10, dtype=torch.float64,
                                                device=dev), x[:, 48:])
    return {"pose": x[:, :48].cpu().numpy(), "trans": x[:, 48:].cpu().numpy(),
            "vertices": verts.cpu().numpy(), "fit_mm": error.cpu().numpy(),
            "iterations": out["iterations"], "device": str(dev)}


class ManoSparse(torch.nn.Module):
    """只算 16 關節 + 指定頂點的 MANO 前向 —— 給 Jacobian 用的輕量版。

    `jacrev` 對完整前向微分要走 778 個頂點的 LBS,但殘差只用到 16 個關節
    與 5 個指尖頂點。實測完整版在 40 幀上跑不完 2 分鐘。

    兩個省法:

    * **betas 固定** → `rest_joints = J_regressor @ v_shaped` 是常數,
      預先算好,不必每次乘 (16,778) 的 dense 矩陣;
    * **只取需要的頂點列** → `v_template`/`shapedirs`/`posedirs`/`weights`
      各只留 5 列,LBS 從 778 降到 5,約 150 倍。

    數值與 `ManoTorch` 相同(同一組公式、同一份陣列),`selftest_sparse()`
    直接比對兩者輸出。
    """

    def __init__(self, model, vertex_ids, device: torch.device,
                 betas=None, dtype=torch.float64):
        super().__init__()
        def t(x):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        ids = list(vertex_ids)
        v_template = t(model.v_template)
        shapedirs = t(model.shapedirs)
        betas_t = (torch.zeros(10, dtype=dtype, device=device)
                   if betas is None else t(betas))
        v_shaped = v_template + torch.einsum("vki,i->vk", shapedirs, betas_t)
        # betas 固定 → rest joints 與 shaped 頂點都是常數
        self.rest = torch.einsum("jv,vk->jk", t(model.J_regressor), v_shaped)
        self.v_shaped = v_shaped[ids]                    # (V,3)
        self.posedirs = t(model.posedirs)[ids]           # (V,3,135)
        self.weights = t(model.weights)[ids]             # (V,16)
        self.hands_mean = t(model.hands_mean)
        self.parents = [int(p) for p in np.asarray(model.parents)]

    def forward(self, pose: torch.Tensor, trans: torch.Tensor,
                *, flat_hand_mean: bool = False):
        """pose (B,48), trans (B,3) → (joints (B,16,3), verts (B,V,3))."""
        B = pose.shape[0]
        if not flat_hand_mean:
            pose = torch.cat([pose[:, :3], pose[:, 3:] + self.hands_mean], 1)
        rot = axis_angle_to_matrix(pose.reshape(B, 16, 3))
        eye = torch.eye(3, dtype=rot.dtype, device=rot.device)
        feature = (rot[:, 1:] - eye).reshape(B, -1)
        posed = self.v_shaped + torch.einsum("vkp,bp->bvk", self.posedirs,
                                             feature)
        global_rot = [rot[:, 0]]
        global_pos = [self.rest[0].expand(B, 3)]
        for j in range(1, 16):
            parent = self.parents[j]
            global_rot.append(global_rot[parent] @ rot[:, j])
            offset = (self.rest[j] - self.rest[parent]).expand(B, 3)
            global_pos.append(
                global_pos[parent]
                + torch.einsum("bik,bk->bi", global_rot[parent], offset))
        R = torch.stack(global_rot, 1)
        P = torch.stack(global_pos, 1)
        offsets = posed.unsqueeze(2) - self.rest.unsqueeze(0).unsqueeze(1)
        transformed = torch.einsum("bjik,bvjk->bvji", R, offsets) + P.unsqueeze(1)
        verts = torch.einsum("vj,bvji->bvi", self.weights, transformed)
        return P + trans.unsqueeze(1), verts + trans.unsqueeze(1)
