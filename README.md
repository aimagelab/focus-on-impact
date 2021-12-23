# Focus on Impact

This is the PyTorch implementation for our paper:

[**Focus on Impact: Indoor Exploration with Intrinsic Motivation**](https://arxiv.org/abs/2109.08521)<br>
__***Roberto Bigazzi***__, Federico Landi, Silvia Cascianelli, Lorenzo Baraldi, Marcella Cornia, Rita Cucchiara<br>

![](content/real_world.gif)

If you find our code useful for your research, please cite our paper:
#### Bibtex:
```
@article{bigazzi2021focus,
  title={Focus on Impact: Indoor Exploration with Intrinsic Motivation},
  author={Bigazzi, Roberto and Landi, Federico and Cascianelli, Silvia and Baraldi, Lorenzo and Cornia, Marcella and Cucchiara, Rita},
  journal={arXiv preprint arXiv:2109.08521},
  year={2021}
}
```

## Table of Contents
   1. [Abstract](#abstract)
   2. [Installation](#installation)
   3. [Training](#training)
   4. [Evaluation](#evaluation)
   5. [Pretrained Models](#pretrained-models)
   6. [Real-World Deployment](#real-world-deployment)
   7. [Acknowledgements](#acknowledgements)

## Abstract

In this work, we propose to train the model with a purely intrinsic reward signal to guide exploration, which is based on the impact of the robot's actions on its internal representation of the environment. So far, impact-based rewards have been employed for simple tasks and in procedurally generated synthetic environments with countable states. Since the number of states observable by the agent in realistic indoor environments is non-countable, we include a neural-based density model and replace the traditional count-based regularization with an estimated pseudo-count of previously visited states.

<p align="center">
<img src="https://fdlandi.github.io/images/focus_on_impact.svg" width="90%"/>
</p>

The proposed exploration approach outperforms DRL-based competitors relying on intrinsic rewards and surpasses the agents trained with a dense extrinsic reward computed with the environment layouts. We also show that a robot equipped with the proposed approach seamlessly adapts to point-goal navigation and real-world deployment.

## Installation

1. Create an environment with *conda*:
    ```
    conda create -n focus_on_imp python=3.6 cmake=3.14.0
    source activate focus_on_imp
    ```

2. Clone this repository:
    ```
    git clone --recursive https://github.com/aimagelab/focus-on-impact
    cd focus-on-impact
    ```
 
3. Install *[Habitat-Lab](https://github.com/facebookresearch/habitat-lab)* and *[Habitat-Sim](https://github.com/facebookresearch/habitat-sim)* in the environments directory:
   ```
   cd environments/habitat-sim
   python -m pip install -r requirements.txt
   python setup.py install --headless --with-cuda
   
   cd ../habitat-lab
   python -m pip install -r requirements.txt
   python -m pip install -r habitat_baselines/rl/requirements.txt
   python setup.py develop --all
   
   cd ../..
   ```

4. Install the requirements for this repository:
   ```
   python -m pip install -r requirements.txt
   ```
    
5. Download the scene datasets *[Gibson](https://github.com/StanfordVL/GibsonEnv#database)* and *[Matterport3D](https://niessner.github.io/Matterport/)* and store them in `data/scene_datasets/`.

6. Download the exploration task datasets for *Gibson* and *Matterport3D* from *[Occupancy Anticipation](https://github.com/facebookresearch/OccupancyAnticipation)* repository and store them in `data/datasets/`.

7. Download the navigation task datasets for *Gibson* from *[Habitat-Lab](https://github.com/facebookresearch/habitat-lab)* repository and store them in `data/datasets/`.
 
8. Install *[A* algorithm](https://github.com/srama2512/astar_pycpp)* used by the Planner:
    ```
    cd occant_utils/astar_pycpp
    make
    cd ../..
    ```
9. Generate environment layout maps as specified in the instruction of *[Occupancy Anticipation](https://github.com/facebookresearch/OccupancyAnticipation)* repository. 
    
## Training
In order to train a model for exploration you can run:

    python -u run.py --exp-config configs/model_configs/{MODEL_TYPE}/ppo_{MODEL_NAME}_training.yaml --run-type train
    
It is possible to change training parameters modifying the *.yaml* file.


## Evaluation
The evaluation of the models can be done both in exploration and pointgoal navigation.

#### Exploration
    python -u run.py --exp-config configs/model_configs/{MODEL_TYPE}/ppo_{MODEL_NAME}_{DATASET}_{SPLIT}_noisy.yaml --run-type eval

#### Navigation
    python -u run.py --exp-config configs/model_configs/{MODEL_TYPE}/ppo_{MODEL_NAME}_navigation.yaml --run-type eval

## Pretrained Models
Pretrained weights of our models on 5 millions total frames.

| Name              | Link                                                                                         |
| ---               | ---                                                                                          |
| Impact (Grid)     | *[Here](https://drive.google.com/file/d/1XpQxB6nZrDVH4C7c2XD9FWNZ7p7f6PMb/view?usp=sharing)* |
| Impact (DME)      | *[Here](https://drive.google.com/file/d/1Kk4js6Dujadws-0FU0ng9pOIohzcySla/view?usp=sharing)* |

## Real-World Deployment
For instructions on the real-world deployment of the models please follow *[LocoNav](https://github.com/aimagelab/LoCoNav)* instructions.

## Acknowledgements
This repository uses parts of *[Habitat-Lab](https://github.com/facebookresearch/habitat-lab)* and *[Habitat-Sim](https://github.com/facebookresearch/habitat-sim)*. We also used *[Occupancy Anticipation](https://github.com/facebookresearch/OccupancyAnticipation)* [1] for some of the competitors, the *[ANS](https://github.com/devendrachaplot/Neural-SLAM)*-based [2] Mapper and the exploration task dataset.

[1] Ramakrishnan S. K., Al-Halah Z., and Grauman K. *"Occupancy anticipation for efficient exploration and navigation"* In ECCV 2020.<br/>
[2] Chaplot D. S., et al. *"Learning to explore using active neural slam"* In ICLR 2020.
