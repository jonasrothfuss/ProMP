[![Build Status](https://api.travis-ci.com/jonasrothfuss/ProMP.svg?branch=master)](https://travis-ci.com/jonasrothfuss/ProMP)
[![Docs](https://readthedocs.org/projects/promp/badge/?version=latest)](https://promp.readthedocs.io)

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

## Installation / Dependencies
The provided code can be either run in A) docker container provided by us or B) using python on
your local machine. The latter requires multiple installation steps in order to setup dependencies.

### A. Docker
If not installed yet, [set up](https://docs.docker.com/install/) docker on your machine.
Pull our docker container ``jonasrothfuss/promp`` from docker-hub:

```
docker pull jonasrothfuss/promp
```

All the necessary dependencies are already installed inside the docker container.

### B. Anaconda or Virtualenv

##### B.1. Installing MPI
Ensure that you have a working MPI implementation ([see here](https://mpi4py.readthedocs.io/en/stable/install.html) for more instructions). 

For Ubuntu you can install MPI through the package manager:

```
sudo apt-get install libopenmpi-dev
```

##### B.2. Create either venv or conda environment and activate it

###### Virtualenv
```
pip install --upgrade virtualenv
virtualenv <venv-name>
source <venv-name>/bin/activate
```

###### Anaconda 
If not done yet, install [anaconda](https://www.anaconda.com/) by following the instructions [here](https://www.anaconda.com/download/#linux).
Then reate a anaconda environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
conda create -n <env-name> python=3.6
source activate <env-name>
```

##### B.3. Install the required python dependencies
```
pip install -r requirements.txt
```

##### B.4. Set up the Mujoco physics engine and mujoco-py
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