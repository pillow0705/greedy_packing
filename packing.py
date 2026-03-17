"""
greedy_packing/packing.py

贪心球堆积算法（无神经网络，无周期边界条件）。

策略：每一步从候选点集合中选择使
    球体总体积 / 最小包围矩形体积
最大的候选点放置新粒子。

候选点生成机制：维护两个增量集合
    _triplet_set:   已有粒子的三元组索引 (i,j,k)，i<j<k
    _candidate_set: 每个三元组产生的候选位置列表

当放置第 n 个粒子后，所有包含该粒子的新三元组被加入 _triplet_set，
并为每个新三元组计算候选点（使新粒子与三元组中三个粒子同时接触）。
"""

import numpy as np
from itertools import combinations
from physics import solve_three_spheres, check_collision, check_single_collision


class GreedyPacking:
    def __init__(self, diameters=(0.8, 1.2), collision_tol=0.02,
                 max_candidates=600, local_radius_factor=3.0):
        """
        Parameters
        ----------
        diameters           : tuple，允许的球直径列表
        collision_tol       : 相对碰撞容差，|gap| < tol * r_sum 视为接触
        max_candidates      : 候选集上限；超出时随机保留，防止 O(N³) 慢化
        local_radius_factor : 局部密度邻域半径 = factor × (r_new + r_avg)
                              值越大越"全局"，越小越局部
        """
        self.diameters           = np.array(diameters)
        self.tol                 = collision_tol
        self.max_candidates      = max_candidates
        self.local_radius_factor = local_radius_factor

        # 粒子状态
        self.positions = []   # list of np.ndarray (3,)
        self.radii     = []   # list of float

        # 增量集合
        self._triplet_set = set()   # frozenset of 3 indices
        self._candidates  = []      # list of dict {position, radius, triplet}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def reset(self):
        """清空所有状态，准备重新堆积。"""
        self.positions = []
        self.radii     = []
        self._triplet_set   = set()
        self._candidates    = []

    def place_first_four(self, tet_radii=None):
        """
        放置初始四面体（4 个球两两接触）。
        tet_radii: 长度为 4 的半径列表；None 则随机选取。
        """
        if tet_radii is None:
            tet_radii = [np.random.choice(self.diameters) / 2.0 for _ in range(4)]

        r = tet_radii
        # 第 1 个球：原点
        p0 = np.array([0.0, 0.0, 0.0])
        # 第 2 个球：沿 x 轴，与 p0 接触
        d01 = r[0] + r[1]
        p1 = np.array([d01, 0.0, 0.0])
        # 第 3 个球：在 xy 平面内，与 p0、p1 同时接触
        d02 = r[0] + r[2]
        d12 = r[1] + r[2]
        x2 = (d02**2 - d12**2 + d01**2) / (2 * d01)
        y2 = np.sqrt(max(d02**2 - x2**2, 0.0))
        p2 = np.array([x2, y2, 0.0])
        # 第 4 个球：用三球解析解
        ok, sol_a, sol_b = solve_three_spheres(p0, r[0], p1, r[1], p2, r[2], r[3])
        if not ok:
            raise ValueError("初始四面体几何无解，请检查半径设置。")
        p3 = sol_a  # 取 z>0 的解（sol_a 对应 +z）

        for p, ri in zip([p0, p1, p2, p3], r):
            self._add_particle(p, ri)

        # 初始四面体的四个三元组全部加入 _triplet_set，并生成候选
        n = len(self.positions)
        for tri in combinations(range(n), 3):
            key = frozenset(tri)
            if key not in self._triplet_set:
                self._triplet_set.add(key)
                self._gen_candidates_for_triplet(tri)

    def step(self):
        """
        执行一步贪心放置。
        从 _candidates 中选择使 phi = 球体积 / 包围盒体积 最大的候选点。

        Returns
        -------
        placed : bool   — 是否成功放置
        phi    : float  — 放置后的体积分数（近似）
        """
        if not self._candidates:
            return False, self._compute_phi()

        best_idx  = -1
        best_phi  = -1.0
        best_cand = None

        # 预计算当前球体积总和（加速）
        cur_sphere_vol = float(np.sum(
            (4.0 / 3.0) * np.pi * np.array(self.radii)**3
        )) if self.radii else 0.0

        for idx, cand in enumerate(self._candidates):
            sol   = cand['position']
            r_new = cand['radius']

            # 碰撞检测
            collision, _ = check_collision(
                sol, r_new, self.positions, self.radii, self.tol
            )
            if collision:
                continue

            # 全局包围盒 phi（加入候选后）
            phi = self._bbox_phi_incremental(sol, r_new, cur_sphere_vol)

            if phi > best_phi:
                best_phi  = phi
                best_idx  = idx
                best_cand = cand

        if best_cand is None:
            # 所有候选碰撞，清空并返回
            self._candidates = []
            return False, self._compute_phi()

        # 放置最优候选
        new_pos = best_cand['position']
        new_rad = best_cand['radius']
        self._add_particle(new_pos, new_rad)

        # 保留不与新粒子碰撞的旧候选（增量过滤），同时去掉刚放置的那个
        kept = []
        for cand in self._candidates:
            if cand is best_cand:
                continue
            col, _ = check_single_collision(
                cand['position'], cand['radius'],
                new_pos, new_rad, self.tol
            )
            if not col:
                kept.append(cand)
        self._candidates = kept

        # 新粒子索引
        new_idx = len(self.positions) - 1

        # 所有包含 new_idx 的新三元组
        for other_two in combinations(range(new_idx), 2):
            tri = tuple(sorted(other_two + (new_idx,)))
            key = frozenset(tri)
            if key not in self._triplet_set:
                self._triplet_set.add(key)
                self._gen_candidates_for_triplet(tri)

        # 候选集过大时随机降采样，防止 O(N³) 慢化
        if len(self._candidates) > self.max_candidates:
            idx = np.random.choice(len(self._candidates),
                                   self.max_candidates, replace=False)
            self._candidates = [self._candidates[i] for i in idx]

        return True, self._compute_phi()

    def run(self, n_steps=200, verbose=True):
        """
        运行贪心堆积 n_steps 步。
        先放置初始四面体，然后循环调用 step()。
        """
        self.reset()
        self.place_first_four()

        history = []
        for step_i in range(n_steps):
            placed, phi = self.step()
            history.append((len(self.positions), phi))
            if verbose and step_i % 20 == 0:
                print(f"  step {step_i:4d} | N={len(self.positions):4d} | "
                      f"phi={phi:.4f} | candidates={len(self._candidates)}")
            if not placed:
                if verbose:
                    print(f"  *** 无可用候选，停止于 step {step_i} ***")
                break

        return history

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @property
    def _centroid(self):
        """当前所有粒子的质心。"""
        if not self.positions:
            return np.zeros(3)
        return np.mean(self.positions, axis=0)

    def _add_particle(self, pos, radius):
        self.positions.append(np.array(pos, dtype=float))
        self.radii.append(float(radius))

    def _gen_candidates_for_triplet(self, tri):
        """
        对三元组 (i,j,k) 计算两个候选点（解析解），
        并为每种可能的新球直径各生成一个候选。
        """
        i, j, k = tri
        p1, r1 = self.positions[i], self.radii[i]
        p2, r2 = self.positions[j], self.radii[j]
        p3, r3 = self.positions[k], self.radii[k]

        for d in self.diameters:
            r_new = d / 2.0
            ok, sol_a, sol_b = solve_three_spheres(p1, r1, p2, r2, p3, r3, r_new)
            if not ok:
                continue
            for sol in (sol_a, sol_b):
                if sol is None:
                    continue
                self._candidates.append({
                    'position': sol,
                    'radius':   r_new,
                    'triplet':  tri,
                })

    def _bbox_phi_incremental(self, sol, r_new, cur_sphere_vol):
        """
        加入候选粒子后的全局包围盒体积分数（增量计算，避免重建 array）。
        """
        new_vol = cur_sphere_vol + (4.0 / 3.0) * np.pi * r_new**3

        pos = np.array(self.positions)
        rad = np.array(self.radii)
        lo  = (pos - rad[:, None]).min(axis=0)
        hi  = (pos + rad[:, None]).max(axis=0)
        lo  = np.minimum(lo, sol - r_new)
        hi  = np.maximum(hi, sol + r_new)
        box_vol = float(np.prod(hi - lo))
        if box_vol < 1e-12:
            return 0.0
        return new_vol / box_vol

    def _compute_phi(self):
        """全局体积分数（用于报告）：球体积之和 / 最小包围盒体积。"""
        if not self.positions:
            return 0.0
        pos = np.array(self.positions)
        rad = np.array(self.radii)
        lo  = (pos - rad[:, None]).min(axis=0)
        hi  = (pos + rad[:, None]).max(axis=0)
        box_vol = float(np.prod(hi - lo))
        if box_vol < 1e-12:
            return 0.0
        return float(np.sum((4.0 / 3.0) * np.pi * rad**3)) / box_vol
