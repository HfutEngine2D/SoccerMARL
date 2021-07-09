#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.a3c import A3CTrainer
from ray.tune import grid_search
from gym import spaces  #space include observation_space and action_space
import numpy as np
import hfo_py
import torch

from soccer_env.mult_agent_env import MultiAgentSoccer
from ray.rllib.models import ModelCatalog
from ray.rllib.examples.models.parametric_actions_model import TorchParametricActionsModel
env_config = {
        "server_config":{
            "defense_npcs": 1,
            "offense_agents":2
        },
        " feature_set": hfo_py.LOW_LEVEL_FEATURE_SET ,
    }

def on_episode_end(info):
    episode = info["episode"]
    # print("episode.last_info_for()", episode.last_info_for(0))
    episode.custom_metrics["goal_rate"] = int(episode.last_info_for(0)['status'] == hfo_py.GOAL)

server_config = env_config["server_config"]
obs_space_size = 59 + 9*(server_config["defense_npcs"]+server_config["offense_agents"]-1)
obs_space = spaces.Box(low=-1, high=1,
                                            shape=((obs_space_size,)), dtype=np.float32)
act_space = spaces.Discrete(14)

def gen_policy(_):
    return (None, obs_space, act_space, {})

# Setup PPO with an ensemble of `num_policies` different policies
policies = {
    'policy_{}'.format(i): gen_policy(i) for i in range(server_config["offense_agents"]) 
}
policy_ids = list(policies.keys())

stop = {
       "timesteps_total": 11000000,
       "episode_reward_mean": 13
       }


results = tune.run(
    PPOTrainer,
    #ImpalaTrainer,
    #A3CTrainer,
     config={
    "env": MultiAgentSoccer,
    "env_config": env_config,
    'multiagent': {
        'policies': policies,
        'policy_mapping_fn':
            lambda agent_id: policy_ids[agent_id],
    },
    "callbacks": {
        "on_episode_end": on_episode_end,
    },
    "lr": grid_search([0.0001]),
    "num_gpus" : torch.cuda.device_count(),
    "num_workers": 5,
    "output": "/tmp/out-soccer", 
    "log_level": "INFO",
    "batch_mode": "complete_episodes",
    "framework": 'torch'
},  checkpoint_freq=100,
    restore="/home/caprlith/ray_results/PPO_2021-07-07_20-30-19/PPO_MultiEnv_177dc_00000_0_lr=0.0001_2021-07-07_20-30-19/checkpoint_2100/checkpoint-2100",
    checkpoint_at_end=True,
    stop=stop)  

# import pickle
# print("best lr:",results.get_best_config(metric="mean_loss",mode="min"))
# fw=open("bestcofig.pkl","wb")
# pickle.dump(results.get_best_config(metric="mean_loss",mode="min"),fw)
# fw.close()