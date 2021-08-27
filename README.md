# A Study of an Open-Ended Strategy for Learning Complex Locomotion Skills

## Description
- Generate diverse and complex three-dimensional training terrains
- The POET-SAC approach combines the Enhanced POET with SAC 

## Requirements
- Python 3.8 
- Pytorch 1.8.1 
- fiber 0.2.1 
- neat-python 0.92 
- gym 0.18.0 
- PyOpenGL 3.1.5
- mujoco 200 
- mujoco_py 2.0 

## Install MuJoCo & mujoco_py
Please refer to https://github.com/openai/mujoco-py

## Installation
```sh
git clone https://github.com/ml-tue/ePOET_3D.git
pip install -r requirements.txt
```

## Add 'torchrl' to python path
```sh
export PYTHONPATH="${PYTHONPATH}:/absolute_dir_to_project/torchrl/"
export PYTHONPATH="${PYTHONPATH}:/absolute_dir_to_project/torchrl/torchrl"
```

## Run training
```sh
./run_poet_local.sh test
```

## Evaluate & Render results
```sh
python learning_curve.py       #plot learning curve
python evaluate_model.py       #evaluate the trained models of POET and POET-SAC
python evaluate_rl_models.py   #evaluate the trained models of PPO, SAC, VMPO
```

## References
[1] Rui Wang, Joel Lehman, Jeff Clune, and Kenneth O. Stanley. Paired open-ended trailblazer (POET): endlessly generating increasingly complex and diverse learning environments and their solutions. CoRR, abs/1901.01753, 2019. 
[2] Rui Wang, Joel Lehman, Aditya Rawal, Jiale Zhi, Yulun Li, Jeff Clune, and Kenneth O. Stanley. Enhanced POET: Open-Ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions, 2020.
https://github.com/uber-research/poet 
[3] Teymur Azayev and Karel Zimmerman. Blind Hexapod Locomotion in Complex Terrain with Gait Adaptation Using Deep Reinforcement Learning and Classification. Journal of Intelligent Robotic Systems, 99, 09 2020.
https://github.com/silverjoda/nexabots 
[4] Rchal Yang. Pytorch implementation of reinforcement learning methods. https://github.com/RchalYang/torchrl
