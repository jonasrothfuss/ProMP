[![Build Status](https://travis-ci.com/jonasrothfuss/maml_zoo.svg?token=j5XSZzvzzLqAy58VEYoC&branch=master)](https://travis-ci.com/jonasrothfuss/maml_zoo)
[![Docs](https://readthedocs.org/projects/promp/badge/?version=latest)](promp.readthedocs.io)
# ProMP: Proximal Meta-Policy Search
Implementations corresponding to ProMP ([Rothfuss et al., 2018](https://arxiv.org/abs/????)). 
Overall this repository consists of two branches:

1) master: lightweight branch that provides the necessary code to run Meta-RL algorithms such as ProMP, E-MAML, MAML.
            This branch is meant to provide an easy start with Meta-RL and can be integrated into other projects and setups.
2) full-code: branch that provides the comprehensive code that was used to produce the experimental results in [Rothfuss et al. (2018)](https://arxiv.org/abs/????).
              This includes experiment scripts and plotting scripts that can be used to reproduce the experimental results in the paper.
              
The code is written in Python 3 and builds on [Tensorflow](https://www.tensorflow.org/). 
Many of the provided reinforcement learning environments require the [Mujoco](http://www.mujoco.org/) physics engine.
Overall the code was developed under consideration of modularity and computational efficiency.
Many components of the Meta-RL algorithm are parallelized either using either [MPI](https://mpi4py.readthedocs.io/en/stable/) 
or [Tensorflow](https://www.tensorflow.org/) in order to ensure efficient use of all CPU cores.

## Getting started

### Installing MPI
Ensure that you have a working MPI implementation ([see here](https://mpi4py.readthedocs.io/en/stable/install.html) for more instructions). 

For Ubuntu you can install MPI through the package manager:

```
sudo apt-get install libopenmpi-dev
```

### Virtualenv
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv <venv-name>
source <venv-name>/bin/activate
pip install -r requirements.txt
```

### Anaconda 
TODO

### Docker container
TODO

### Setting up mujoco-py
For running the majority of the provided Meta-RL environments, the Mujoco physics engine as well as a 
corresponding python wrapper are required.
For setting up [Mujoco](http://www.mujoco.org/) and [mujoco-py](https://github.com/openai/mujoco-py), 
please follow the instructions [here](https://github.com/openai/mujoco-py).



## Running ProMP
In order to run the ProMP algorithm with default configurations execute:
```
python run_scripts/pro-mp_run.py 
```

The run configuration can be change either in the run script directly or by providing a JSON configuration file with all
the necessary hyperparameters. A JSON configuration file can be provided through the flag. Additionally the dump path 
can be specified through the dump_path flag:

```
python run_scripts/pro-mp_run.py --config_file <config_file_path> --dump_path <dump_path>
```

## Acknowledgements
This repository includes environments introduced in ([Duan et al., 2016](https://arxiv.org/abs/1611.02779), 
[Finn et al., 2017](https://arxiv.org/abs/1703.03400)).