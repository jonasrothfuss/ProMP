import sys
sys.path.append('.')
import os
import os.path as osp
import argparse
import ast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, default=None, nargs='?')
    parser.add_argument('--dry', action='store_true', default=False)
    parser.add_argument('--bare', action='store_true', default=False)
    args = parser.parse_args()
    remote_dir = "s3://rllab-experiments/doodad/logs"
    local_dir = osp.abspath(osp.join(osp.dirname(__file__), '../data/s3'))
    if args.folder:
        remote_dir = osp.join(remote_dir, args.folder)
        local_dir = osp.join(local_dir, args.folder)
    if args.bare:
        command = ("""
            aws s3 sync {remote_dir} {local_dir} --exclude '*' --include '*.csv' --include '*.json' --content-type "UTF-8"
        """.format(local_dir=local_dir, remote_dir=remote_dir))
    else:
        command = ("""
            aws s3 sync {remote_dir} {local_dir} --exclude '*stdout.log' --exclude '*stdouterr.log' --content-type "UTF-8"
        """.format(local_dir=local_dir, remote_dir=remote_dir))
    if args.dry:
        print(command)
    else:
        os.system(command)