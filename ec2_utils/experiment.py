# Copied from doodad/run_experiment_lite_doodad.py
import os
import pickle
import base64
import argparse

ARGS_DATA = 'DOODAD_ARGS_DATA'
USE_CLOUDPICKLE = 'DOODAD_USE_CLOUDPICKLE'
CLOUDPICKLE_VERSION = 'DOODAD_CLOUDPICKLE_VERSION'

__ARGS = None
def __get_arg_config():
    """
    global __ARGS
    if __ARGS is not None:
        return __ARGS
    #TODO: use environment variables rather than command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cloudpickle', type=bool, default=False)
    parser.add_argument('--'+ARGS_DATA, type=str, default='')
    parser.add_argument('--output_dir', type=str, default='/tmp/expt/')
    args = parser.parse_args()
    __ARGS = args
    """
    args_data = os.environ.get(ARGS_DATA, {})
    cloudpickle_version = os.environ.get(CLOUDPICKLE_VERSION, 'n/a')
    use_cloudpickle = bool(int(os.environ.get(USE_CLOUDPICKLE, '0')))

    args = lambda : None # hack - use function as namespace
    args.args_data = args_data
    args.use_cloudpickle = use_cloudpickle
    args.cloudpickle_version = cloudpickle_version
    return args

def get_args(key=None, default=None):
    args = __get_arg_config()

    if args.args_data:
        if args.use_cloudpickle:
            import cloudpickle
            assert args.cloudpickle_version == cloudpickle.__version__, "Cloudpickle versions do not match! (host) %s vs (remote) %s" % (args.cloudpickle_version, cloudpickle.__version__)
            data = cloudpickle.loads(base64.b64decode(args.args_data))
        else:
            data = pickle.loads(base64.b64decode(args.args_data))
    else:
        data = {}

    if key is not None:
        return data.get(key, default)
    return data

def encode_args(call_args, cloudpickle=False):
    """
    Encode call_args dictionary as a base64 string
    """
    assert isinstance(call_args, dict)

    if cloudpickle:
        import cloudpickle
        cpickle_version = cloudpickle.__version__
        data = base64.b64encode(cloudpickle.dumps(call_args)).decode("utf-8")
    else:
        data = base64.b64encode(pickle.dumps(call_args)).decode("utf-8")
        cpickle_version = 'n/a'
    return data, cpickle_version

# These are arguments passed in from launch_python
args_dict = get_args()
print('My args are:', args_dict)

