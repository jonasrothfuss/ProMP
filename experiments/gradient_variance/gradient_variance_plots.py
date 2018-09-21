import matplotlib.pyplot as plt
from experiment_utils import plot_utils
import numpy as np

plt.style.use('ggplot')
#plt.rc('font', family='Times New Roman')
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.font_manager._rebuild()


SMALL_SIZE = 30
MEDIUM_SIZE = 32
BIGGER_SIZE = 36
LINEWIDTH = 4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
COLORS = dict(ours=colors.pop(0))

LEGEND_ORDER = {'VPG': 0, 'DICE': 1}

########## Add data path here #############
data_path = '/home/jonasrothfuss/Dropbox/Eigene_Dateien/UC_Berkley/2_Code/maml-zoo/data/s3/gradient-variance'
###########################################
exps_data = plot_utils.load_exps_data([data_path])

def sorting_legend(label):
    return LEGEND_ORDER[label]


def get_color(label):
    if label not in COLORS.keys():
        COLORS[label] = colors.pop(0)
    return COLORS[label]

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


def plot_from_exps(exp_data,
                   filters={},
                   split_plots_by=None,
                   x_key='n_timesteps',
                   y_keys=None,
                   plot_name='./plot',
                   subfigure_titles=None,
                   plot_labels=None,
                   x_label=None,
                   y_labels=None,
                   running_mean_size=5,
                   cut_x_range_at=None
                   ):

    exp_data = plot_utils.filter(exp_data, filters=filters)
    exps_per_plot = dict([(y_key, exp_data) for y_key in y_keys])
    fig, axarr = plt.subplots(1, len(exps_per_plot.keys()), figsize=(20, 6))
    fig.tight_layout(pad=2., w_pad=1.5, h_pad=3.0, rect=[0, 0, 0.9, 1])

    if not hasattr(axarr, '__iter__'): axarr = [axarr]

    # x axis formatter
    xfmt = matplotlib.ticker.ScalarFormatter()
    xfmt.set_powerlimits((3, 3))

    # iterate over subfigures
    for i, (y_key, plot_exps) in enumerate(sorted(exps_per_plot.items())):
        plots_in_figure_exps = plot_utils.group_by(plot_exps, split_plots_by)
        subfigure_title = subfigure_titles[i] if subfigure_titles else y_key
        axarr[i].set_title(subfigure_title)
        axarr[i].xaxis.set_major_formatter(xfmt)
        axarr[i].xaxis.set_major_locator(plt.MaxNLocator(5))

        # iterate over plots in figure
        for j, default_label in enumerate(sorted(plots_in_figure_exps, key=sorting_legend)):
            exps = plots_in_figure_exps[default_label]
            x, y_mean, y_std = plot_utils.prepare_data_for_plot(exps, x_key=x_key, y_key=y_key, round_x=True)

            if 'Gradient' in y_key:
                y_mean = runningMeanFast(y_mean, running_mean_size)
                y_std = runningMeanFast(y_std, running_mean_size)

            if cut_x_range_at is not None:
                x, y_mean, y_std = x[:cut_x_range_at], y_mean[:cut_x_range_at], y_std[:cut_x_range_at]

            label = plot_labels[j] if plot_labels else default_label
            _label = label if i == 0 else "__nolabel__"
            axarr[i].plot(x, y_mean, label=_label,  linewidth=LINEWIDTH, color=get_color(label))
            axarr[i].fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2, color=get_color(label))

            # axis labels
            axarr[i].set_xlabel(x_label if x_label else x_key)
            axarr[i].set_ylabel(y_labels[i] if y_labels[i] else y_key)

            # axis ticks

    axarr[0].set_ylim(0, 80)


    fig.legend(loc='center right', ncol=1, bbox_transform=plt.gcf().transFigure)
    fig.savefig(plot_name + '.pdf')

plot_from_exps(exps_data,
               filters={'normalize_by_path_length': False},
               split_plots_by='algo',
               x_key='n_timesteps',
               y_keys=['Meta-GradientRStd', 'Step_1-AverageReturn'],
               x_label='Time steps',
               y_labels=['Relative Std', 'Average Return'],
               subfigure_titles=['Gradient Variance', 'Return'],
               plot_labels=['LCV', 'DICE'],
               plot_name='./gradient_variance',
               cut_x_range_at=295
               )