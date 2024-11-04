# AI2P-WindFarmControl

This repository contains the code accompanying the "Real-Time Wake Steering Control for Wind Farms using
Graph-based Modeling and Deep Reinforcement Learning" paper.

The code has been implemented in Python version 3.11.
The used packages can be found in the [requirements.txt](requirements.txt) file and can be installed
using pip by running the command: <pre> ```pip install -r requirement.txt``` </pre>


- The [data](data) folder gives and overview of the expected file structure for the input data. 
- The [architecture](architecture) folder contains the PIGNN, LSTM and UNet architectures employed in the experiments.
- The [utils](utils) folder contains several utility functions used for data preprocessing, visualization and evaluation.
- The [graphs](experiments/graphs) folder contains the code for the graph-based wind speed map generation experiments. We first used [prepare_training_data.py](prepare_training_data.py) to translate the input data into precomputed graphs that could be used by the PIGNN. The experiments were ran through the [graph_experiments.py](graph_experiments.py) file, and we employed the [graph_evaluation.ipynb](graph_evaluation.ipynb) notebook to evaluate the results, which can be found in the [results](experiments/graphs/results) folder.
- The [reinforcement_learning](experiments/reinforcement_learning) folder contains the code for the reinforcement learning experiments. The [generation_model](experiments/reinforcement_learning/generation_model) folder contains the pretrained PIGNN 30 model for Wake Steering Case 1. A custom [environment](experiments/reinforcement_learning/env_continuous.py) was used to train and evaluate the [SAC](experiments/reinforcement_learning/sac.py), [TD3](experiments/reinforcement_learning/td3.py), and [DDPG](experiments/reinforcement_learning/ddpg.py) agents. We employed [eval_continuous.py](experiments/reinforcement_learning/eval_continuous.py) to evaluate the results, which can be found in the [results](experiments/reinforcement_learning/results) folder.