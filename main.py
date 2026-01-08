import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

weights_normal = {
    "uniform": 0.4,  # 均勻分布
    "center": 1.2,  # 中心高斯分布
    "anti": 1.1,  # 反焦點分布
    "border": 1.3,  # 邊緣偏好分布
    "levelk": 0.7,  # 水平混合分布
}

weights_new = {
    "uniform": 0.50,  # 很多人剛來會亂選或隨便點
    "center": 1.40,  # 視覺/直覺中心偏好高
    "anti": 0.80,  # 會避中心的人較少
    "border": 1.00,  # 邊界有吸引力但不極端
    "levelk": 0.60,  # 深度推理者很少
}


weights_old = {
    "uniform": 0.05,  # 幾乎不亂選
    "center": 0.70,  # 中心被視為高競爭區（反而下降）
    "anti": 2.50,  # 強烈避開大家會預期的選擇（反焦點）
    "border": 1.80,  # 邊界/角落變成常見替代
    "levelk": 2.00,  # 很多人會做多層次推理
}

weights_mix = {
    "uniform": (weights_new["uniform"] + weights_old["uniform"]) / 2,
    "center": (weights_new["center"] + weights_old["center"]) / 2,
    "anti": (weights_new["anti"] + weights_old["anti"]) / 2,
    "border": (weights_new["border"] + weights_old["border"]) / 2,
    "levelk": (weights_new["levelk"] + weights_old["levelk"]) / 2,
}

weights_all = {
    "normal": weights_normal,
    "new": weights_new,
    "old": weights_old,
    "mix": weights_mix,
}


# ---------- 工具函數 ----------
def normalize(v):
    v = np.array(v, dtype=float)
    return v / v.sum()


# ---------- 核心模型 ----------
def compute_distribution(width, height, weights):
    coords = [(x, y) for y in range(1, height + 1) for x in range(1, width + 1)]

    cx, cy = (width + 1) / 2, (height + 1) / 2
    sx, sy = width / 4, height / 4

    # 1. Center Gaussian
    gauss = np.array(
        [
            np.exp(-(((x - cx) ** 2) / (2 * sx * sx) + ((y - cy) ** 2) / (2 * sy * sy)))
            for x, y in coords
        ]
    )
    gauss = normalize(gauss)

    # 2. Uniform
    uniform = np.ones(len(coords)) / len(coords)

    # 3. Anti-focal
    anti = normalize(1 - gauss)

    # 4. Border favor
    border = np.array(
        [
            2.0 if (x == 1 or x == width or y == 1 or y == height) else 0.9
            for x, y in coords
        ]
    )
    border = normalize(border)

    # 5. Level-k mix
    describable = np.array(
        [
            (
                1.0
                if (
                    x in (1, width)
                    or y in (1, height)
                    or abs(x - cx) < 1
                    or abs(y - cy) < 1
                )
                else 0.0
            )
            for x, y in coords
        ]
    )
    non_desc = 1.0 - describable
    levelk = normalize(
        0.45 * gauss + 0.35 * border + 0.20 * (non_desc + 0.1 * describable)
    )

    # ---------- 加權整合 ----------

    agg = (
        weights["uniform"] * uniform
        + weights["center"] * gauss
        + weights["anti"] * anti
        + weights["border"] * border
        + weights["levelk"] * levelk
    )

    agg = normalize(agg)
    return coords, agg


# ---------- 畫熱圖 ----------
def plot_heatmap(width, height, weights_name, weights):
    coords, probs = compute_distribution(width, height, weights)

    grid = np.zeros((height, width))
    for (x, y), p in zip(coords, probs):
        grid[y - 1, x - 1] = p

    plt.figure(figsize=(width / 2, height / 2))
    plt.imshow(grid, origin="lower")
    plt.colorbar(label="Selection Probability")
    plt.title(f"{width} x {height} Selection Heatmap")
    plt.xticks(range(width), range(1, width + 1))
    plt.yticks(range(height), range(1, height + 1))
    plt.tight_layout()

    filename = f"{weights_name}_{width}x{height}_heatmap.png"
    plt.savefig(filename, dpi=200)
    plt.close()

    # 冷點推薦
    df = pd.DataFrame(
        {"x": [c[0] for c in coords], "y": [c[1] for c in coords], "prob": probs}
    ).sort_values("prob")

    print(f"\n=== {width} x {height} 最低被選機率 Top 10 ===")
    print(df.head(10).assign(prob=lambda d: (d.prob * 100).round(3)))

    return filename


# ---------- 執行 ----------
for w, h in [(8, 5), (10, 12), (16, 10)]:
    for weights_name, weights in weights_all.items():
        fname = plot_heatmap(w, h, weights_name, weights)
    print(f"已輸出熱圖：{fname}")
