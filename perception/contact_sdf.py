"""幾何接觸估計:物體表面距離 + softmin —— 依 arXiv 2410.16571 §IV-B。

## 為什麼

現行 `extract_golden.compute_contact_heuristic` 只吃手部關節速度與屈曲度,
**完全不看物體在哪**(實測物體相關引用次數 = 0),雖然每支 clip 都有網格與
GT 位姿。已量到的後果:`taco_20230928_032` 工具手接觸 0/190、三支 clip
左手拇指全 0。

Yang et al.(arXiv 2410.16571)的作法把接觸表達成手表面點到物體 SDF 的
softmin 加權:

    hⱼ = exp(−δ·φ(pⱼ,o)) / Σₖ exp(−δ·φ(pₖ,o))     (eq. 15)
    Φᵢ = Σⱼ hⱼ·φ(pⱼ,o)          手指 i 到物體的距離   (eq. 16)
    cᵢ = Σⱼ hⱼ·pⱼ               接觸點               (eq. 17)
    nᵢ = Σⱼ hⱼ·∂φ/∂pⱼ           接觸法向             (eq. 18)

關鍵在 **cᵢ 會隨姿態在手表面上移動** —— 這是滾動能被表達的前提,而用固定的
指尖關節點做不到(實測:以關節為接觸點時,滑動量與接觸點位移在數學上恆等,
見 `eval/rolling_vs_sliding.py` 的更正)。

## 兩個明講的近似

* **無號距離**。論文用 SDF(有號);此處用到表面的無號距離。接觸判定只需
  φ→0,無號足夠,且不必做內外測試。要做穿透量則需補上號。
* **點對頂點近似點對表面**。TACO 物體網格 21K–27K 頂點,間距遠小於接觸
  尺度,所以用 KD-tree 找最近頂點是可控的近似。`surface_spacing_mm()`
  會報出實際間距,讓這個近似可被檢驗而不是被假設。

    python -m perception.contact_sdf --clip taco_20231020_254 --side right
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# MANO 16 關節 → 五指。0=wrist, 1-3 index, 4-6 middle, 7-9 pinky,
# 10-12 ring, 13-15 thumb(見 perception/mano_layer.py 檔頭)。
FINGER_JOINTS = {
    "thumb": (13, 14, 15), "index": (1, 2, 3), "middle": (4, 5, 6),
    "ring": (10, 11, 12), "little": (7, 8, 9),
}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
SOFTMIN_TEMPERATURE = 200.0   # δ,單位 1/m。論文建議高溫使 softmin 逼近真 min


@dataclass
class ContactEstimate:
    distance_mm: np.ndarray      # (5,)  Φᵢ,每指到物體的 softmin 距離
    point: np.ndarray            # (5,3) cᵢ,世界系接觸點(會在手表面移動)
    normal: np.ndarray           # (5,3) nᵢ,單位化的接觸法向


def load_object_surface(path: Path, max_points: int = 20000) -> np.ndarray:
    """讀 `_cm.obj` 頂點(cm → m)。網格夠密時頂點即可代表表面。"""
    vertices = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    array = np.asarray(vertices, dtype=np.float64) / 100.0
    if len(array) > max_points:                # 抽稀,間距仍遠小於接觸尺度
        array = array[:: len(array) // max_points + 1]
    return array


def surface_spacing_mm(points: np.ndarray, sample: int = 2000) -> float:
    """表面取樣間距中位數 —— 點對頂點近似的誤差尺度,報出來供檢驗。"""
    tree = cKDTree(points)
    index = np.random.default_rng(0).choice(
        len(points), size=min(sample, len(points)), replace=False)
    distance, _ = tree.query(points[index], k=2)
    return float(np.median(distance[:, 1])) * 1000.0


def finger_vertex_groups(weights: np.ndarray) -> dict[str, np.ndarray]:
    """用蒙皮權重把 778 個頂點分到五指(取權重最大的關節所屬手指)。"""
    owner = np.argmax(weights, axis=1)
    groups = {}
    for finger, joints in FINGER_JOINTS.items():
        groups[finger] = np.where(np.isin(owner, joints))[0]
    return groups


def contact_from_surface(hand_vertices: np.ndarray,
                         object_points_world: np.ndarray,
                         groups: dict[str, np.ndarray],
                         temperature: float = SOFTMIN_TEMPERATURE,
                         tree: cKDTree | None = None) -> ContactEstimate:
    """單幀:每指的 (Φᵢ, cᵢ, nᵢ)。公式見檔頭 eq. 15-18。

    `tree` 是**純選用**的加速路徑:物體位姿在一幀之內固定,但最佳化器會在
    同一幀評估上百次殘差,每次重建 20K 點的 KD-tree 要 ~30ms —— 那會讓
    `retarget/contact_synth.py` 的每幀求解花 6 秒以上。傳進來的樹必須是
    `cKDTree(object_points_world)`,不然量到的距離對應的是別的物體位姿而
    **不會報錯**。預設 None 時行為與先前逐字相同。
    """
    tree = cKDTree(object_points_world) if tree is None else tree
    distance = np.zeros(5)
    point = np.zeros((5, 3))
    normal = np.zeros((5, 3))
    for k, finger in enumerate(FINGER_ORDER):
        index = groups[finger]
        points = hand_vertices[index]
        phi, nearest = tree.query(points, k=1)
        # softmin:低溫會把遠點也算進來,高溫逼近真 min。權重同時決定
        # 距離、接觸點與法向,所以三者一致(論文的可微性來源)。
        shifted = -temperature * (phi - phi.min())
        weight = np.exp(shifted)
        weight /= weight.sum()
        distance[k] = float(weight @ phi)
        point[k] = weight @ points
        # ∂φ/∂p 的方向 = 由最近表面點指向查詢點(無號距離的梯度)
        direction = points - object_points_world[nearest]
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = np.divide(direction, norm, out=np.zeros_like(direction),
                              where=norm > 1e-9)
        vector = weight @ direction
        length = np.linalg.norm(vector)
        normal[k] = vector / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])
    return ContactEstimate(distance_mm=distance * 1000.0, point=point,
                           normal=normal)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import h5py

    from perception.mano_layer import load_mano

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clip", default="taco_20231020_254")
    parser.add_argument("--side", default="right", choices=("left", "right"))
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args(argv)

    directory = PROJECT_ROOT / "data" / "golden" / args.clip
    with h5py.File(directory / "original_metadata.hdf5", "r") as handle:
        ids = [o.decode() for o in handle["object_ids"][:]]
        roles = {}
        for obj in ids:
            key = f"object_role/{obj}"
            if key in handle:
                raw = handle[key][()]
                roles[obj] = raw.decode() if isinstance(raw, bytes) else str(raw)
        held = next((o for o, r in roles.items()
                     if (r == "tool") == (args.side == "right")), ids[0])
        pose = handle[f"object_pose/{held}"][:].astype(np.float64)

    surface = load_object_surface(directory / "objects" / f"{held}_cm.obj")
    spacing = surface_spacing_mm(surface)
    model = load_mano(args.side)
    groups = finger_vertex_groups(model.weights)

    print(f"clip {args.clip} side {args.side} held {held}")
    print(f"  物體表面點 {len(surface)},取樣間距中位數 {spacing:.2f} mm")
    print(f"  每指頂點數 " + ", ".join(
        f"{f}:{len(groups[f])}" for f in FINGER_ORDER))
    print("  (完整逐幀估計需要手部網格,見 perception/contact_pipeline.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
