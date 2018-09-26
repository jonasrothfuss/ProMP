import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys
import joblib
import tensorflow as tf
import pickle


itrs = [11, 0, 21]
exp_names = ['lr', 'll_noexp', 'll_exp']
plot_names = ['LVC', 'MAML', 'E-MAML']
fig, axarr = plt.subplots(1, 3, figsize=(8, 3))
fig.tight_layout()
# fig.tight_layout(pad=2, w_pad=.25, h_pad=0.5, rect=[0, 0, 1, 1])
fig.subplots_adjust(wspace=0.2, hspace=0)
for k, (i, exp_name, plotname) in enumerate(zip(itrs, exp_names, plot_names)):
    with open("sandbox/" + exp_name + '.pkl', 'rb') as filename:
        all_samples_data = pickle.load(filename)

    ax = axarr[k]
    ax.set_title(plotname)
    ax.set(adjustable='box-forced', aspect='equal')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    goal = plt.Circle((2, 2), 2, color='g', alpha=0.2, zorder=-2)
    goal2 = plt.Circle((2, 2), 0.3, color='g', zorder=-1)
    f1 = plt.Circle((-2, -2), 2, color='b', alpha=0.2, zorder=-2)
    f2 = plt.Circle((-2, 2), 2, color='b', alpha=0.2, zorder=-2)
    f3 = plt.Circle((2, -2), 2, color='b', alpha=0.2, zorder=-2)
    ax.add_artist(goal)
    ax.add_artist(goal2)
    ax.add_artist(f1)
    ax.add_artist(f2)
    ax.add_artist(f3)
    # color_list = ['Blues', 'Purples', 'Reds', 'Oranges', 'YlGn', 'Greens']
    color_list = ['blue', 'purple', 'red', 'orange']
    labels = ['Pre-update', 'Step 1', 'Step 2', 'Post-update']
    for step, (paths, color, label) in enumerate(zip(all_samples_data, color_list, labels)):
        if step == 1 or step == 2:
           continue
        all_obses = paths[i]['observations']
        for j in range(10,20): # plot 5 trajectories per task
            obs_range = list(range(j * 100, (j+1) * 100))
            obses = all_obses[obs_range]
            colors = np.arange(len(obses))
            if k == 0 and j == 10:
                ax.scatter(obses[:,0], obses[:,1], c=color, label=label, s=3)
            else:
                ax.scatter(obses[:,0], obses[:,1], c=color, s=3)

handles, labels = axarr[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_transform=plt.gcf().transFigure, markerscale=3, fontsize=14)
plt.show()