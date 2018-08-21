import csv
import os
import numpy as np
import json
import itertools

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def unique(l):
    return list(set(l))


def flatten(l):
    return [item for sublist in l for item in sublist]


def load_progress(progress_csv_path):
    print("Reading %s" % progress_csv_path)
    entries = dict()
    with open(progress_csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k, v in row.items():
                if k not in entries:
                    entries[k] = []
                try:
                    entries[k].append(float(v))
                except:
                    entries[k].append(0.)
    entries = dict([(k, np.array(v)) for k, v in entries.items()])
    return entries


# def to_json(stub_object):
#     from rllab.misc.instrument import StubObject
#     from rllab.misc.instrument import StubAttr
#     if isinstance(stub_object, StubObject):
#         assert len(stub_object.args) == 0
#         data = dict()
#         for k, v in stub_object.kwargs.items():
#             data[k] = to_json(v)
#         data["_name"] = stub_object.proxy_class.__module__ + \
#                         "." + stub_object.proxy_class.__name__
#         return data
#     elif isinstance(stub_object, StubAttr):
#         return dict(
#             obj=to_json(stub_object.obj),
#             attr=to_json(stub_object.attr_name)
#         )
#     return stub_object


def flatten_dict(d):
    flat_params = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = flatten_dict(v)
            for subk, subv in flatten_dict(v).items():
                flat_params[k + "." + subk] = subv
        else:
            flat_params[k] = v
    return flat_params


def load_params(params_json_path):
    with open(params_json_path, 'r') as f:
        data = json.loads(f.read())
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-2]
    return data


def lookup(d, keys):
    if not isinstance(keys, list):
        keys = keys.split(".")
    for k in keys:
        if hasattr(d, "__getitem__"):
            if k in d:
                d = d[k]
            else:
                return None
        else:
            return None
    return d


def load_exps_data(exp_folder_paths, disable_variant=False):
    exps = []
    for exp_folder_path in exp_folder_paths:
        exps += [x[0] for x in os.walk(exp_folder_path)]
    exps_data = []
    for exp in exps:
        try:
            exp_path = exp
            params_json_path = os.path.join(exp_path, "params.json")
            variant_json_path = os.path.join(exp_path, "variant.json")
            progress_csv_path = os.path.join(exp_path, "progress.csv")
            progress = load_progress(progress_csv_path)
            if disable_variant:
                params = load_params(params_json_path)
            else:
                try:
                    params = load_params(variant_json_path)
                except IOError:
                    params = load_params(params_json_path)
            exps_data.append(AttrDict(
                progress=progress, params=params, flat_params=flatten_dict(params)))
        except IOError as e:
            print(e)
    return exps_data


def smart_repr(x):
    if isinstance(x, tuple):
        if len(x) == 0:
            return "tuple()"
        elif len(x) == 1:
            return "(%s,)" % smart_repr(x[0])
        else:
            return "(" + ",".join(map(smart_repr, x)) + ")"
    else:
        if hasattr(x, "__call__"):
            return "__import__('pydoc').locate('%s')" % (x.__module__ + "." + x.__name__)
        else:
            return repr(x)


def extract_distinct_params(exps_data, excluded_params=('exp_name', 'seed', 'log_dir'), l=1):
    try:
        stringified_pairs = sorted(
            map(
                eval,
                unique(
                    flatten(
                        [
                            list(
                                map(
                                    smart_repr,
                                    list(d.flat_params.items())
                                )
                            )
                            for d in exps_data
                        ]
                    )
                )
            ),
            key=lambda x: (
                tuple(0. if it is None else it for it in x),
            )
        )
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()
    proposals = [(k, [x[1] for x in v])
                 for k, v in itertools.groupby(stringified_pairs, lambda x: x[0])]
    filtered = [(k, v) for (k, v) in proposals if len(v) > l and all(
        [k.find(excluded_param) != 0 for excluded_param in excluded_params])]
    return filtered


class Selector(object):
    def __init__(self, exps_data, filters=None, custom_filters=None):
        self._exps_data = exps_data
        if filters is None:
            self._filters = tuple()
        else:
            self._filters = tuple(filters)
        if custom_filters is None:
            self._custom_filters = []
        else:
            self._custom_filters = custom_filters

    def where(self, k, v):
        return Selector(self._exps_data, self._filters + ((k, v),), self._custom_filters)

    def custom_filter(self, filter):
        return Selector(self._exps_data, self._filters, self._custom_filters + [filter])

    def _check_exp(self, exp):
        # or exp.flat_params.get(k, None) is None
        return all(
            ((str(exp.flat_params.get(k, None)) == str(v) or (k not in exp.flat_params)) for k, v in self._filters)
        ) and all(custom_filter(exp) for custom_filter in self._custom_filters)

    def extract(self):
        return list(filter(self._check_exp, self._exps_data))

    def iextract(self):
        return filter(self._check_exp, self._exps_data)


# Taken from plot.ly
color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]


def hex_to_rgb(hex, opacity=1.0):
    if hex[0] == '#':
        hex = hex[1:]
    assert (len(hex) == 6)
    return "rgba({0},{1},{2},{3})".format(int(hex[:2], 16), int(hex[2:4], 16), int(hex[4:6], 16), opacity)
