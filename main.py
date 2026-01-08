import os
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PATH_OUTPUT = "output"
NUM_WORKERS = 28

if not os.path.exists(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT)

# ---------- 權重設定 ----------
weights_normal = {
    "uniform": 0.4,
    "center": 1.2,
    "anti": 1.1,
    "border": 1.3,
    "levelk": 0.7,
}

weights_new = {
    "uniform": 0.50,
    "center": 1.40,
    "anti": 0.80,
    "border": 1.00,
    "levelk": 0.60,
}

weights_old = {
    "uniform": 0.05,
    "center": 0.70,
    "anti": 2.50,
    "border": 1.80,
    "levelk": 2.00,
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
    s = v.sum()
    if s == 0:
        return np.ones_like(v) / len(v)
    return v / s


# ---------- 核心模型（加噪聲） ----------
def compute_distribution(
    width, height, weights, noise_level=0.6, gauss_share=0.8, use_noise=True
):
    coords = [(x, y) for y in range(1, height + 1) for x in range(1, width + 1)]
    n = len(coords)

    cx, cy = (width + 1) / 2, (height + 1) / 2
    sx, sy = width / 4, height / 4

    # Center Gaussian
    gauss = np.array(
        [
            np.exp(-(((x - cx) ** 2) / (2 * sx * sx) + ((y - cy) ** 2) / (2 * sy * sy)))
            for x, y in coords
        ]
    )
    gauss = normalize(gauss)

    # Uniform
    uniform = np.ones(n) / n

    # Anti-focal
    anti = normalize(1 - gauss)

    # Border favor
    border = np.array(
        [
            2.0 if (x == 1 or x == width or y == 1 or y == height) else 0.9
            for x, y in coords
        ]
    )
    border = normalize(border)

    # Level-k mix
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

    # 加權整合
    agg = (
        weights["uniform"] * uniform
        + weights["center"] * gauss
        + weights["anti"] * anti
        + weights["border"] * border
        + weights["levelk"] * levelk
    )
    agg = normalize(agg)

    if not use_noise:
        return coords, agg, 0.0, 0.0

    # 噪聲設定
    mean_p = 1.0 / n
    noise_magnitude = noise_level * mean_p
    gauss_std = noise_magnitude * gauss_share
    uniform_half_range = noise_magnitude * (1.0 - gauss_share)

    return coords, agg, gauss_std, uniform_half_range


# ---------- 大量模擬 ----------
def simulate_many(
    width,
    height,
    weights,
    weights_name="",
    n_iter=10_000_000,
    noise_level=0.6,
    gauss_share=0.8,
    use_noise=True,
    pbar=None,
):
    coords, base_probs, gauss_std, uniform_half_range = compute_distribution(
        width, height, weights, noise_level, gauss_share, use_noise
    )
    n = len(coords)
    counts = np.zeros(n, dtype=np.float64)

    batch_size = 100_000
    n_batches = n_iter // batch_size
    for b in range(n_batches):
        if use_noise:
            gauss_noise = np.random.normal(0, gauss_std, size=(batch_size, n))
            uniform_noise = np.random.uniform(
                -uniform_half_range, uniform_half_range, size=(batch_size, n)
            )
            total_noise = gauss_noise + uniform_noise
            probs_batch = base_probs + total_noise
            probs_batch = np.clip(probs_batch, 1e-12, None)
            probs_batch = probs_batch / probs_batch.sum(axis=1, keepdims=True)
        else:
            probs_batch = np.tile(base_probs, (batch_size, 1))

        samples = np.random.choice(
            n, size=batch_size, p=probs_batch[0] if not use_noise else None
        )
        if use_noise:
            for i in range(batch_size):
                s = np.random.choice(n, p=probs_batch[i])
                counts[s] += 1
        else:
            for s in samples:
                counts[s] += 1

        if pbar is not None:
            pbar.update(batch_size)
            pbar.set_postfix_str(f"{width}x{height} [{weights_name}]")

    remaining = n_iter % batch_size
    if remaining > 0:
        if use_noise:
            gauss_noise = np.random.normal(0, gauss_std, size=(remaining, n))
            uniform_noise = np.random.uniform(
                -uniform_half_range, uniform_half_range, size=(remaining, n)
            )
            total_noise = gauss_noise + uniform_noise
            probs_batch = base_probs + total_noise
            probs_batch = np.clip(probs_batch, 1e-12, None)
            probs_batch = probs_batch / probs_batch.sum(axis=1, keepdims=True)
            for i in range(remaining):
                s = np.random.choice(n, p=probs_batch[i])
                counts[s] += 1
        else:
            samples = np.random.choice(n, size=remaining, p=base_probs)
            for s in samples:
                counts[s] += 1
        if pbar is not None:
            pbar.update(remaining)

    probs_final = counts / counts.sum()
    return coords, probs_final


def run_single_task(task_params):
    width, height, weights, weights_name, n_iter, noise_params = task_params
    coords, probs = simulate_many(
        width,
        height,
        weights,
        weights_name=weights_name,
        n_iter=n_iter,
        pbar=None,
        **noise_params,
    )
    filename = plot_heatmap_from_probs(
        width, height, coords, probs, weights_name, n_iter
    )
    return filename


# ---------- 畫熱圖 ----------
def plot_heatmap_from_probs(
    width, height, coords, probs, weights_name, total_iterations
):
    grid = np.zeros((height, width))
    for (x, y), p in zip(coords, probs):
        grid[y - 1, x - 1] = p

    plt.figure(figsize=(width / 2, height / 2))
    plt.imshow(grid, origin="lower")
    plt.colorbar(label="Selection Probability")
    plt.title(f"{width} x {height} Selection Heatmap ({weights_name})")
    plt.xticks(range(width), range(1, width + 1))
    plt.yticks(range(height), range(1, height + 1))
    plt.tight_layout()

    filename = f"{weights_name}_{width}x{height}_heatmap.png"
    grid_dir = os.path.join(PATH_OUTPUT, f"{total_iterations:,}")
    if not os.path.exists(grid_dir):
        os.makedirs(grid_dir)
    plt.savefig(os.path.join(grid_dir, filename), dpi=200)
    plt.close()
    return filename


# ---------- 命令列介面 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate human-like selection with noise."
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        required=True,
        help="Number of simulation iterations",
    )
    parser.add_argument(
        "-n",
        "--noise",
        action="store_true",
        help="Enable Gaussian + Uniform noise",
    )
    args = parser.parse_args()

    noise_params = {"noise_level": 0.6, "gauss_share": 0.8, "use_noise": args.noise}

    grid_sizes = [(8, 5), (10, 12), (16, 10)]
    total_tasks = len(grid_sizes) * len(weights_all)
    total_iterations = total_tasks * args.iterations

    logging.info(
        f"開始模擬: {total_tasks} 個任務, 共 {total_iterations:,} 次迭代, 使用 {NUM_WORKERS} 個 workers"
    )
    start_time = time.time()

    tasks = []
    for w, h in grid_sizes:
        for weights_name, weights in weights_all.items():
            tasks.append((w, h, weights, weights_name, args.iterations, noise_params))

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_single_task, task): task for task in tasks}
        with tqdm(total=total_tasks, desc="圖片生成進度", unit="圖") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    filename = future.result()
                    logging.info(f"已輸出熱圖：{filename}")
                except Exception as e:
                    logging.error(f"任務失敗 {task[3]} {task[0]}x{task[1]}: {e}")
                pbar.update(1)

    elapsed = time.time() - start_time
    speed = total_iterations / elapsed
    logging.info(f"完成！總耗時: {elapsed:.2f} 秒, 平均速度: {speed:,.0f} iter/s")
