#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 14:27:58 2018

@author: qiutian
"""

import numpy as np
import os
import scipy.stats as st
import matplotlib.pyplot as plt


###############################################################################
### re-scale the bottom part of the y-axis
def bottom_scale(arr, scale=[-200, -60, -50], dim=None):
    if scale is None:
        return arr
    if dim is None:
        arr = np.array(arr).reshape(-1)
        arr_n = np.zeros(arr.shape)
        ratio = (scale[2] - scale[1]) / (scale[2] - scale[0])

        def shrink(num):
            if num < scale[2]:
                return scale[2] - ratio * (scale[2] - num)
            else:
                return num

        for idx, ii in enumerate(arr):
            arr_n[idx] = shrink(ii)
        return arr_n
    else:
        arr_n = []
        for row in arr:
            arr_n.append(bottom_scale(row, scale=scale, dim=None))
        return np.array(arr_n)


p_output = 'output/navi_v1'


###############################################################################
### Only plot the Fine-tune performance curve
def plot_performance():
    ic = range(0, 10)
    num = 200
    mark = 10

    # Only keep Fine-tune data reading
    rews_finetune_0 = np.load(os.path.join(p_output, 'rewards.npy'))[ic, :num]

    # Calculate mean and standard error
    def ave_rews_stats(arr):
        ave_rews = np.mean(arr, axis=0)
        ave_ave_rews = np.mean(ave_rews)
        err_ave_rews = np.std(ave_rews, ddof=1) / np.sqrt(len(ave_rews))
        return ave_ave_rews, err_ave_rews

    mean_val, err_val = ave_rews_stats(rews_finetune_0)
    print('rewards, mean: %.3f, standard error: %.3f' % (mean_val, err_val))

    # Calculate 95% confidence interval
    def conf_int(arr):
        arr_stats = np.zeros((3, arr.shape[1]))
        for idx in range(arr.shape[1]):
            col = arr[:, idx]
            arr_stats[0, idx] = np.mean(col)
            down, up = st.t.interval(0.95, len(col) - 1, loc=np.mean(col),
                                     scale=st.sem(col))
            arr_stats[1, idx], arr_stats[2, idx] = down, up
        return arr_stats

    rews_finetune = conf_int(rews_finetune_0)

    # Y-axis scaling
    scale = [-200, -60, -50]
    rews_finetune = bottom_scale(rews_finetune, scale=scale, dim=1)

    # Plot settings
    plt.figure(figsize=(4, 3), dpi=200)
    alpha = 0.1
    ms = 4
    lw = 1
    mew = 1
    tick_size = 8
    label_size = 10
    x = np.arange(1, num + 1)

    # Only plot the Fine-tune curve
    plt.fill_between(x, rews_finetune[1], rews_finetune[2], color='darkorange', alpha=alpha)
    plt.plot(x, rews_finetune[0], color='darkorange', lw=lw,
             marker='^', markevery=mark, ms=ms, mew=mew, mfc='white')

    # Legend
    plt.legend(['rewards'], labelspacing=0.1, fancybox=True, shadow=True, fontsize=label_size)

    plt.xlabel('Learning Episodes', fontsize=label_size)
    plt.ylabel('Return', fontsize=label_size)
    plt.xticks(np.arange(0, num + 1, num // 5), fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.grid(axis='y', ls='-', lw=0.2)
    plt.grid(axis='x', ls='-', lw=0.2)

    plt.axis([0, num, -60, 2])
    yticks = [-200, -50, -40, -30, -20, -10, 0]
    plt.yticks(np.arange(-60, 10, 10), yticks)


# Execute plotting
plot_performance()
plt.show()