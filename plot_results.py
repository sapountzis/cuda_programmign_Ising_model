import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


if __name__ == '__main__':
    stats_file = 'stats.csv'
    plot_size = (35, 25)
    font_size = 70

    if stats_file not in os.listdir():
        raise FileNotFoundError

    stats = pd.read_csv(stats_file).sort_values(by='n', axis=0)
    sizes = stats['n'].unique()
    implementations = stats['implementation'].unique()
    x_axis = np.arange(len(sizes))

    font = {'weight': 'bold', 'size': font_size}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.set_size_inches(plot_size)

    [i.set_linewidth(7) for i in ax.spines.values()]

    ax.tick_params(which='major', axis='x', labelsize=font_size, width=10, length=40, direction='out',
                   pad=20)
    ax.tick_params(which='major', axis='y', labelsize=font_size, width=10, length=40, direction='out',
                   pad=20)

    legend_patches = []

    for i, implementation in enumerate(implementations):
        stats_sel = stats[stats['implementation'] == implementation]
        avg_loop_time_log = np.log10(stats_sel['avg_loop_time'])

        ax.plot(x_axis, avg_loop_time_log, linewidth=7, color=f'C{i}')
        legend_patches.append(mpatches.Patch(color=f'C{i}', label=implementation))

    ax.set_xlabel('n', fontweight='bold', labelpad=20, fontsize=font_size)
    ax.set_ylabel('Avg Loop Time (log10 sec)', fontweight='bold', labelpad=20, fontsize=font_size)

    ax.set_xticks(x_axis)
    ax.set_xticklabels(sizes)

    ax.legend(handles=legend_patches, handlelength=1, frameon=False)

    fig.tight_layout()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va='center')
    fig.tight_layout()

    fig.savefig('stats.png')


