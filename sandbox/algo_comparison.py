from viskit import core
import matplotlib.pyplot as plt
from sandbox.plot_utils_ppo import *

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
COLORS = dict()

LEGEND_ORDER = {'ppo (ours)': 0, 'trpoll': 1, 'e-trpoll':2, 'vpglr': 3, 'e-vpgll': 4}
# 'trpolr': 2, 'vpgll': 3, 
def sorting_legend(label):
    return LEGEND_ORDER[label]

def get_color(label):
    if label not in COLORS.keys():
        COLORS[label] = colors.pop(0)
    return COLORS[label]

def plot_from_exps(exp_data,
                   filters={},
                   split_figures_by=None,
                   split_plots_by=None,
                   x_key='n_timesteps',
                   y_key=None,
                   plot_name='./algo-comparison',
                   subfigure_titles=None,
                   plot_labels=None,
                   x_label=None,
                   y_label=None,
                   ):

    exp_data = filter(exp_data, filters=filters)
    exps_per_plot = group_by(exp_data, group_by_key=split_figures_by)
    fig, axarr = plt.subplots(height, len(exps_per_plot.keys()) // height, figsize=(40, 10))
    fig.tight_layout(pad=4., w_pad=1.5, h_pad=3.0, rect=[0, 0, 1, 1])

    # x axis formatter
    xfmt = matplotlib.ticker.ScalarFormatter()
    # xfmt.set_powerlimits((3, 3))

    # iterate over subfigures
    for i, (default_plot_title, plot_exps) in enumerate(sorted(exps_per_plot.items())):
      plots_in_figure_exps = group_by(plot_exps, split_plots_by)
      subfigure_title = subfigure_titles[i] if subfigure_titles else default_plot_title
      # i = (i % height, i // height)
      axarr[i].set_title(subfigure_title)
      axarr[i].xaxis.set_major_formatter(xfmt)
      axarr[i].xaxis.set_major_locator(plt.MaxNLocator(5))

      # iterate over plots in figure
      length = 1000
      for j, default_label in enumerate(sorted(plots_in_figure_exps, key=sorting_legend)):
          exps = plots_in_figure_exps[default_label]
          x, y_mean, y_std = prepare_data_for_plot(exps, x_key=x_key, y_key=y_key)
          label = plot_labels[j] if plot_labels else default_label
          _label = label
          axarr[i].plot(x, y_mean, label=_label,  linewidth=LINEWIDTH, color=get_color(label))
          axarr[i].fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2, color=get_color(label))
          if len(x) < length:
            length = len(x)
          # axis labels
      # axarr[i].set_xlim((1, length))
      # axarr[i].set_xscale('log')
      axarr[i].set_xlabel(x_label if x_label else x_key)
      axarr[i].set_ylabel(y_label if y_label else y_key)
    handles, labels = axarr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=7, bbox_transform=plt.gcf().transFigure)
    fig.savefig(plot_name + '.pdf')

def add_itr(exp_data):
  for exp in exp_data:
    if 'Itr' not in exp['progress'] and len(exp['progress']) > 0:
      exp['progress']['Itr'] = np.arange(len(exp['progress']['Time']))
    if len(exp['progress']) > 0:
    # if 'n_timesteps' not in exp['progress'] and len(exp['progress']) > 0:
      steps_per_itr = exp['flat_params']['meta_batch_size'] * exp['flat_params']['rollouts_per_meta_task'] * exp['flat_params']['max_path_length'] * (exp['flat_params']['num_inner_grad_steps'] + 1)
      exp['progress']['n_timesteps'] = exp['progress']['Itr'] * steps_per_itr
    
def handle_sawyer(exp_data):
  for exp in exp_data:
    if 'Sawyer' in exp['flat_params']['env.$class']:
      exp['progress']['Step_1-AverageReturn'] = exp['progress']['Step_1-AveragePlaceDistance']
      if 'exp_tag' in exp['flat_params'] and 'Medium' in exp['flat_params']['exp_tag']:
        exp['flat_params']['env.$class'] += 'Medium'
    if 'Momentum' in exp['flat_params']['env.$class']:
      exp['progress']['Step_1-AverageReturn'] = exp['progress']['Step_3-AverageReturn']
    if 'RandGoal' in exp['flat_params']['env.$class']:
      exp['progress']['Step_1-AverageReturn'] = exp['progress']['Step_2-AverageReturn']

def add_exp_name(exp_data, name):
  for exp in exp_data:
    if name == 'ppo':
      exp['flat_params']['exp_name'] = 'ppo (ours)'
    elif 'inner_type' in exp['flat_params'] and 'log' in exp['flat_params']['inner_type']:
      exp['flat_params']['exp_name'] = name + 'll'
    elif 'inner_type' in exp['flat_params'] and 'ratio' in exp['flat_params']['inner_type']:
      exp['flat_params']['exp_name'] = name + 'lr'
    else:
      exp['progress'] = {} # Delete
    if 'exploration' in exp['flat_params'] and exp['flat_params']['exploration']:
      exp['flat_params']['exp_name'] = 'e-' + exp['flat_params']['exp_name']

def clean(exp_data, env_classs):
  cleaned_exp_data = []
  for exp in exp_data:
    if 'Itr' not in exp['progress'] or 'exp_name' not in exp['flat_params']:
      continue
    elif exp['flat_params']['exp_name'] not in LEGEND_ORDER:
      continue
    else:
      for c in env_classes:
        if c in exp['flat_params']['env.$class']:
          cleaned_exp_data.append(exp)
  return cleaned_exp_data

def process(exp_data):
  add_itr(exp_data)
  handle_sawyer(exp_data)
  return clean(exp_data, env_classes)

env_classes = ['HalfCheetah', 'AntRandDirec2D', 'Walker2DRandParams', 'HumanoidRandDirec2D']
height = 1

exps_data = []

data_path = 'data/all-envs-trpo'
exp_data = core.load_exps_data([data_path], False)
add_exp_name(exp_data, 'trpo')
exp_data = process(exp_data)
exps_data.extend(exp_data)

data_path = 'data/all-envs-vpg'
exp_data = core.load_exps_data([data_path], False)
add_exp_name(exp_data, 'vpg')
exp_data = process(exp_data)
exps_data.extend(exp_data)

data_path = "data/all-envs-ppo"
exp_data = core.load_exps_data([data_path], False)
add_exp_name(exp_data, 'ppo')
exp_data = process(exp_data)
exps_data.extend(exp_data)
# filters = dict()

# exp_data = filter(exp_data, filters=filters)


plot_from_exps(exps_data,
               split_figures_by='env.$class',
               split_plots_by='exp_name',
               y_key='Step_1-AverageReturn',
               subfigure_titles=['AntRandDir', 'HalfCheetahFwdBack', 'HumanoidRandDir', 'WalkerRandParams'],
               x_label='Timesteps',
               y_label='Average return',
               )