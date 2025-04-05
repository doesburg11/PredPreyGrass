## Getting started with the PettingZoo/SB3 framework


<p align="center">
    <img src="../../assets/images/readme/predpreygrass.png" width="700" height="80"/> 
    assets/images/readme/predpreygrass.png
</p>


/home/doesburg/Dropbox/03_marl_code/PredPreyGrass/assets/images/readme/predpreygrass.png


### Visualize a random policy with the PettingZoo/SB3 solution
In Visual Studio Code run:
```predpreygrass/pettingzoo/eval/evaluate_random_policy.py```
</br>
<p align="center">
    <img src="../../assets/images/gifs/predpreygrass_random.gif" width="1000" height="200"/>
</p>


### Training and visualize trained model using PPO from stable baselines3

Adjust parameters accordingly in:

[```predpreygrass/pettingzoo/config/config_predpreygrass.py```](hhttps://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/config/config_predpreygrass.py)

In Visual Studio Code run:

[```predpreygrass/pettingzoo/train/train_sb3_ppo_parallel_wrapped_aec_env.py```](hhttps://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/train/train_sb3_ppo_parallel_wrapped_aec_env.py)

To evaluate and visualize after training follow instructions in:

[```predpreygrass/pettingzoo/eval/evaluate_ppo_from_file_aec_env.py```](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/eval/evaluate_ppo_from_file_aec_env.py)

Batch training and evaluating in one go:

[```predpreygrass/pettingzoo/eval/parameter_variation_train_wrapped_to_parallel_and_evaluate_aec.py```](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/eval/parameter_variation_train_wrapped_to_parallel_and_evaluate_aec.py)
