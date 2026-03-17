"""
greedy_packing/run.py

运行贪心球堆积并可视化结果。

用法：
    python3 run.py [--steps N] [--seed S] [--no-plot]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无显示器时使用非交互后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from packing import GreedyPacking


def plot_spheres(positions, radii, title="Greedy Packing", save_path="packing.png"):
    """绘制所有球的 3-D 散点图（球心 + 半径映射到散点大小）。"""
    pos = np.array(positions)
    rad = np.array(radii)

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection='3d')

    colors = plt.cm.viridis(rad / rad.max())
    # 散点大小与半径成正比（像素^2），仅做示意
    sizes  = (rad / rad.max() * 300).astype(float)

    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               s=sizes, c=colors, alpha=0.6, edgecolors='none')

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  图像已保存至 {save_path}")


def plot_history(history, save_path="phi_history.png"):
    """绘制 phi（体积分数）随粒子数的增长曲线。"""
    Ns   = [h[0] for h in history]
    phis = [h[1] for h in history]

    plt.figure(figsize=(7, 4))
    plt.plot(Ns, phis, lw=1.5)
    plt.xlabel("粒子数 N")
    plt.ylabel("体积分数 φ")
    plt.title("Greedy Packing — φ vs N")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  历史曲线已保存至 {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Greedy sphere packing (no NN, no PBC)")
    parser.add_argument("--steps", type=int, default=300,
                        help="最大放置步数（默认 300）")
    parser.add_argument("--seed",  type=int, default=42,
                        help="随机种子（默认 42）")
    parser.add_argument("--no-plot", action="store_true",
                        help="跳过绘图（无 matplotlib 环境时使用）")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"=== Greedy Packing  steps={args.steps}  seed={args.seed} ===")

    packer = GreedyPacking(diameters=(0.8, 1.2), collision_tol=0.02)
    history = packer.run(n_steps=args.steps, verbose=True)

    final_N   = len(packer.positions)
    final_phi = packer._compute_phi()
    print(f"\n最终结果: N={final_N}  φ={final_phi:.4f}")

    if not args.no_plot:
        plot_spheres(packer.positions, packer.radii,
                     title=f"Greedy Packing  N={final_N}  φ={final_phi:.4f}")
        plot_history(history)
    else:
        print("  (绘图已跳过)")


if __name__ == "__main__":
    main()
