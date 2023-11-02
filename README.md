This code release accompanies the following paper:

### [Efficient Q-Learning over Visit Frequency Maps for Multi-agent Exploration of Unknown Environments](https://arxiv.org/abs/2307.16318)*

Xuyang Chen, Ashvin N. Iyer, Zixing Wang, Ahmed H. Qureshi

*IROS 2023, Detroit, Michigan

## Installation

We recommend using a [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment for this codebase. The following commands will set up a new conda environment with the correct requirements (tested on Ubuntu 20.04 LTS):

> Note: Other versions of `torch` and `cudatoolkit` may work but are not tested. You may need to modify the packages in `environment.yml` to work for your system configuration. 

```bash
# Create and activate new conda env and install packages
conda env create -n ivfm --file=environment.yml
conda activate ivfm

# Install shortest path module (used in simulation environment)
cd spfa
python setup.py install
```

## Quickstart

You can  use `enjoy.py` to run a trained policy in the simulation environment.

For example, to load a pretrained policy, you can run:

```bash
python enjoy.py --config-path config/opt-sam-opt/config.yml
```

## Training in the Simulation Environment

The [`config`](config) directory contains template config files for all experiments in the paper. To start a training run, you can give one of the template config files to the `train.py` script. For example, the following will train an I-VFM policy with one agent on a regular sized arena:

```
python train.py config/opt-sam-opt.yml
```

The training script will create a log directory and checkpoint directory for the new training run inside `logs/` and `checkpoints/`, respectively. Inside the log directory, it will also create a new config file called `config.yml`, which stores training run config variables and can be used to resume training or to load a trained policy for evaluation.


### Simulation Environment

To explore the simulation environment using our proposed dense action space (spatial action maps), you can use the `tools_click_agent.py` script, which will allow you to click on the local overhead map to select actions and move around in the environment.

```bash
python tools_click_agent.py
```

### Evaluation

Trained policies can be evaluated using the `evaluate.py` script, which takes in the config path for the training run. For example, to evaluate the SAM-VFM (B) pretrained policy, you can run:

```
python evaluate.py --config-path logs/sam-36000/config.yml --use_gui --show_state_representation --show_occupancy_map
```

This will load the trained policy from the specified training run, and run evaluation on it. The results are saved to an `.npy` file in the `eval` directory. You can then modify the visualize.py to open the .npy file and visualize it.

## Acknowledgement

We'd like to thank the authors of the paper [**Spatial Action Maps Augmented with Visit Frequency Maps for Exploration Tasks**](https://ieeexplore.ieee.org/abstract/document/9636813/) for sharing their code and method.
