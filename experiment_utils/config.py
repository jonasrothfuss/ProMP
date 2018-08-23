import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DOCKER_MOUNT_DIR = '/root/code/data'

DOCKER_IMAGE = 'jonasrothfuss/maml-zoo'

S3_BUCKET_NAME = 'maml-zoo-experiments'