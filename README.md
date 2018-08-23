[![Build Status](https://travis-ci.com/jonasrothfuss/maml_zoo.svg?token=j5XSZzvzzLqAy58VEYoC&branch=master)](https://travis-ci.com/jonasrothfuss/maml_zoo)

# MAML-ZOO

Different implementations of Model-Agnostic Meta-Learning (MAML) applied on Reinforcement Learning problems. This repository includes environments introduced in ([Duan et al., 2016](https://arxiv.org/abs/1611.02779), [Finn et al., 2017](https://arxiv.org/abs/1703.03400)): multi-armed bandits, tabular MDPs, continuous control with MuJoCo, and 2D navigation task.

## Generating the documentation
Run
```
make clean && make html
```
in the docs/ directory to generate the html documentation.
The html are saved in docs/build/

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
You can use the [`main.py`](main.py) script in order to run reinforcement learning experiments with MAML. This script was tested with Python 3.5. Note that some environments may also work with Python 2.7 (all experiments besides MuJoCo-based environments).
```
python main.py --env-name HalfCheetahDir-v1 --num-workers 8 --fast-lr 0.1 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 1000 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-halfcheetah-dir --device cuda
```

## Setting up the EC2 experiment pipeline 

Clone the doodad repository 

```
git clone https://github.com/jonasrothfuss/doodad.git
```

Install the extra package requirements for doodad
```
cd doodad && pip install -r requirements.txt
```

Configure doodad for your ec2 account. First you have to specify the following environment variables in your ~/.bashrc: 
AWS_ACCESS_KEY, AWS_ACCESS_KEY, DOODAD_S3_BUCKET

Then run
```
python scripts/setup_ec2.py
```

Set S3_BUCKET_NAME in experiment_utils/config.py to your bucket name