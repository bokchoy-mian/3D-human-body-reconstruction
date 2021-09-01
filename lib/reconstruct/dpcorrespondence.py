#!/usr/bin/python
#-- coding:utf8 --
import numpy as np
import math
from lib.reconstruct.Helper import distance

def dist2d(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def dpBoundarymatch( smpl_bound,rgb_bound, k):


    rgb_len = rgb_bound.shape[0]
    smpl_len = smpl_bound.shape[0]
    dpValue = np.zeros((rgb_len, smpl_len))
    dpMatch = np.zeros((rgb_len, smpl_len), dtype=np.int)

    dpValue = dpValue + float('+inf')

    for j in range(smpl_len):
        dpValue[0][j] = dist2d(rgb_bound[0], smpl_bound[j])
        dpMatch[0][j] = j

    tmp_dpValue = np.zeros(k+1)

    for i in range(1, rgb_len):
        for j in range(0, smpl_len):
            lastpoint_idx = dpMatch[i-1][j]
            for m in range(0, k+1):
                if lastpoint_idx + m >= smpl_len:
                    lastpoint_idx_mod = lastpoint_idx - smpl_len
                else:
                    lastpoint_idx_mod = lastpoint_idx

                tmp_dpValue[m] = dist2d(rgb_bound[i], smpl_bound[lastpoint_idx_mod + m])

            dpValue[i][j] = dpValue[i-1][j] + np.min(tmp_dpValue)
            dpMatch[i][j] = lastpoint_idx + np.where(tmp_dpValue == np.min(tmp_dpValue))[0][0]
            if dpMatch[i][j] >= smpl_len:
                dpMatch[i][j] -= smpl_len

        # print('%d/%d dp bound match' % (i, rgb_len))

    min_j = np.where(dpValue[rgb_len-1] == np.min(dpValue[rgb_len-1]))
    # print('the best correlation is: %d' % min_j[0][0])
    phi = dpMatch[:, min_j]
    phi = phi[:, 0, 0]

    return phi


def boundary_match(smpl_bound, rgb_bound, k):
    """动态规划求解边界对应点

    :param smpl_bound:
    :param rgb_bound:
    :param k:
    :return:
    """
    smpl_len = smpl_bound.shape[0]
    rgb_len = rgb_bound.shape[0]
    dp = []

    min = 9999
    tmp = []
    for i in range (smpl_len):
        dist = distance (rgb_bound[0], smpl_bound[i])
        if (dist < min):
            min = dist
            tmp.append ((min, i))
        else:
            tmp.append (tmp[-1])
    dp.append (tmp)

    for i in range (1, rgb_len):
        tmp = []
        prev = dp[i - 1]
        p = rgb_bound[i]
        tmp.append ((9999, -1))
        for j in range (1, smpl_len):
            d1 = tmp[j - 1][0]
            prev = dp[i - 1][j - 1]
            if (j > prev[1] and j - prev[1] <= k):
                dist = distance (p, smpl_bound[j])
                d2 = dist + prev[0]
                if (d1 < d2):
                    tmp.append (tmp[-1])
                else:
                    tmp.append ((d2, j))
            else:
                tmp.append (tmp[-1])
        # print(tmp[-1])
        dp.append (tmp)

    match = []
    idx = smpl_len - 1
    for i in range (rgb_len - 1, -1, -1):
        idx = dp[i][idx][1]
        match = [idx] + match
        idx -= 1

    return match