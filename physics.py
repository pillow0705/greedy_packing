import numpy as np


def diff(p1, p2):
    return p1 - p2


def solve_three_spheres(p1, r1, p2, r2, p3, r3, r_new):
    dp21 = p2 - p1
    d12  = np.linalg.norm(dp21)
    s1, s2, s3 = r1 + r_new, r2 + r_new, r3 + r_new

    if d12 > s1 + s2 or d12 < abs(s1 - s2):
        return False, None, None

    ex   = dp21 / d12
    dp31 = p3 - p1
    i    = np.dot(ex, dp31)
    ey_v = dp31 - i * ex
    d_ey = np.linalg.norm(ey_v)

    if d_ey < 1e-7:
        return False, None, None
    ey = ey_v / d_ey
    ez = np.cross(ex, ey)

    x    = (s1**2 - s2**2 + d12**2) / (2 * d12)
    y    = (s1**2 - s3**2 + i**2 + d_ey**2) / (2 * d_ey) - (i * x) / d_ey
    z_sq = s1**2 - x**2 - y**2

    if z_sq < 0:
        return False, None, None
    z = np.sqrt(z_sq)

    return True, p1 + x*ex + y*ey + z*ez, p1 + x*ex + y*ey - z*ez


def check_collision(sol, r_new, all_pos, all_rad, tol):
    """返回 (collision, coordination)。无 PBC。"""
    coordination = 0
    for m in range(len(all_pos)):
        r_sum = all_rad[m] + r_new
        gap   = np.linalg.norm(all_pos[m] - sol) - r_sum
        if gap < -tol * r_sum:
            return True, 0
        if gap < tol * r_sum:
            coordination += 1
    return False, coordination


def check_single_collision(sol, r_new, new_pos, new_rad, tol):
    """返回 (collision, touching)。无 PBC。"""
    r_sum = new_rad + r_new
    gap   = np.linalg.norm(new_pos - sol) - r_sum
    return gap < -tol * r_sum, gap < tol * r_sum
