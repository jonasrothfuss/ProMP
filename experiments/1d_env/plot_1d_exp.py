import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

plt.style.use('ggplot')
#plt.rc('font', family='Times New Roman')
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.font_manager._rebuild()


SMALL_SIZE = 26
MEDIUM_SIZE = 26
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

path_expl = '' # Path of the E-MAML policy
path_dice = '' # Path of the dice policy
LINEWIDTH = 1.25
WIDTH = 0.0075
SIZE = 150
SCALE = 2
ALPHA = 0.8


def process_thetas(thetas, frequency, scale, min_val, max_val):
    vector = scale * (np.array(thetas[min_val+1:max_val]) - np.array(thetas[min_val:max_val-1]))
    ss_thetas = thetas[min_val:max_val-1:int(frequency)]
    ss_vects = list(vector)[::int(frequency)]

    x, y = list(zip(*ss_thetas))
    u, v = list(zip(*ss_vects))
    return x, y, u, v

def get_colors(x, color_map):
    cmap = matplotlib.cm.get_cmap(color_map)
    z = np.arange(len(x))
    z_scaled = (z - np.min(z))/(np.max(z) - np.min(z)) * 0.8 + 0.3
    colors = [cmap(value) for value in z_scaled]
    return colors


def plot_thetas(ax, thetas_dice, thetas_exp, frequency=1, scale=1., min_val=None, max_val=None):
    if max_val is None:
        max_val = len(thetas_dice)
    if min_val is None:
        min_val = 0

    x, y, u, v = process_thetas(thetas_exp, frequency, scale, min_val, max_val)
    colors = get_colors(x, "Greens")
    ax.scatter(x, y, color=colors, linewidth=LINEWIDTH, edgecolors='k', label='II-VPG', s=SIZE)
    ax.quiver(x, y, u, v, color=colors, width=WIDTH, angles='xy', scale_units='xy', scale=1, headwidth=2, headlength=4, linewidth=LINEWIDTH/3, alpha=ALPHA,  edgecolors='k')

    x, y, u, v = process_thetas(thetas_dice, frequency, scale, min_val, max_val)
    colors = get_colors(x, "Reds")
    ax.scatter(x, y, color=colors, linewidth=LINEWIDTH, edgecolors='k', label='I-VPG', s=SIZE)
    ax.quiver(x, y, u, v, color=colors, width=WIDTH, angles='xy', scale_units='xy', scale=1, headwidth=2, headlength=4, linewidth=LINEWIDTH/3, alpha=ALPHA,  edgecolors='k')

    ax.scatter([0], [0], color='gold', marker='^', s=500, linewidth=LINEWIDTH, edgecolors='k', label="Optimum")
    ax.legend()
    ax.set_ylabel(r'$\theta_1$')
    ax.set_xlabel(r'$\theta_0$')

    ax.set_xlim([-.015, 0.1])
    ax.set_ylim([-.01, 0.05])

    plt.xticks(np.arange(-0.015, 0.1, 0.04))
    plt.yticks(np.arange(-0.01, 0.05, 0.02))





if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.tight_layout(pad=3.5, rect=[0, 0, 1, 1])
    thetas_exp = joblib.load(os.path.join(path_expl, 'thetas.pkl'))
    thetas_dice = joblib.load(os.path.join(path_dice, 'thetas.pkl'))
    plot_thetas(ax, thetas_dice, thetas_exp, frequency=3, scale=SCALE, min_val=0, max_val=150)
    plt.title("Meta-Policy Updates")
    fig.savefig('') # Path where to save the figure
