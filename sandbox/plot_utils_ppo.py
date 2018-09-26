import numpy as np
from collections import OrderedDict, defaultdict

def filter(exps_data, filters={}):
    print("before filtering", len(exps_data), 'exps')
    keep_array = []
    if filters:
        for i, exp in enumerate(exps_data):
            keep_array.append(all([((filter_key not in exp['flat_params']) or ((filter_key in exp['flat_params']) and (exp['flat_params'][filter_key] == filter_val)))
                                              for filter_key, filter_val in filters.items()]))
        exps_data_filtered = np.array(exps_data)
        exps_data_filtered = exps_data_filtered[keep_array]
    else:
        exps_data_filtered = exps_data
    print("after filtering", len(exps_data_filtered), 'exps')
    return exps_data_filtered

def group_by(exp_data, group_by_key=None):
    split_dict = OrderedDict()
    for exp in exp_data:
        key_str = str(exp['flat_params'][group_by_key])
        if key_str in split_dict.keys():
            split_dict[key_str].append(exp)
        else:
            split_dict[key_str] = [exp]
    return split_dict

def prepare_data_for_plot(exp_data, x_key='n_timesteps', y_key=None, sup_y_key=None, round_x=None):
    x_y_tuples = []
    for exp in exp_data:
        name = exp['flat_params']['exp_name'].replace('-', '_')
        off_set = 0
        if sup_y_key is not None:
            assert type(sup_y_key) is list
            for key in sup_y_key:
                if key in exp['progress'].keys():
                    x_y_tuples.extend(list(zip(exp['progress'][x_key]-off_set, exp['progress'][key])))
                    break
        else:
            x_y_tuples.extend(list(zip(exp['progress'][x_key], exp['progress'][y_key])))
    x_y_dict = defaultdict(list)
    for k, v in x_y_tuples:
        if round_x is not None:
            x_y_dict[(k//round_x) * round_x].append(v)
        else:
            x_y_dict[k].append(v)
    means, stddevs = [], []
    for key in sorted(x_y_dict.keys()):
        means.append(np.mean(x_y_dict[key]))
        stddevs.append(np.std(x_y_dict[key]))
    return np.array(sorted(x_y_dict.keys())), np.array(means), np.array(stddevs)


def prepare_data_for_plot_all(exp_data, x_key='n_timesteps', y_key=None, sup_y_key=None, round_x=None):
    x_y_tuples = []
    for exp in exp_data:
        name = exp['flat_params']['exp_name'].replace('-', '_')
        key_str = str(name).split('_')[2]
        if key_str == 'maml':
            off_set = (exp['progress'][x_key][1] - exp['progress'][x_key][0])/2
        else:
            off_set = 0
        if sup_y_key is not None:
            assert type(sup_y_key) is list
            for key in sup_y_key:
                if key in exp['progress'].keys():
                    x_y_tuples.append((exp['progress'][x_key]-off_set, exp['progress'][key]))
                    break
        else:
            x_y_tuples.append((exp['progress'][x_key] - off_set, exp['progress'][y_key]))
    return x_y_tuples


def correct_limit(ax, x, y):
   # ax: axes object handle
   #  x: data for entire x-axes
   #  y: data for entire y-axes
   # assumption: you have already set the x-limit as desired
   lims = ax.get_xlim()
   i = np.where((x > lims[0]) &  (x < lims[1]))[0]
   return y[i].min(), y[i].max()
