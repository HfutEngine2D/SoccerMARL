#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.agents.sac import SACTrainer
from ray.tune import grid_search
from gym import spaces  #space include observation_space and action_space
import numpy as np
import hfo_py
import torch

from soccer_env.mult_agent_env import MultiAgentSoccer
from ray.rllib.models import ModelCatalog
from models.parametric_actions_model import TorchParametricActionsModel

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
origin_obs_space = spaces.Box(low=-1, high=1,
                                            shape=((obs_space_size,)), dtype=np.float32)
observation_space=spaces.Dict({
            "action_mask":spaces.Box(0,1,shape=(14,),dtype=np.float32),
            # "avail_actions":spaces.Box(-10,10,shape=(14,2),dtype=np.float32),   #todo
            "orgin_obs":origin_obs_space

        })
act_space = spaces.Discrete(14)

def gen_policy(_):
    return (None, observation_space, act_space, {})

# Setup PPO with an ensemble of `num_policies` different policies
policies = {
    'policy_{}'.format(i): gen_policy(i) for i in range(server_config["offense_agents"]) 
}
policy_ids = list(policies.keys())

stop = {
       "timesteps_total": 10000000,
       "episode_reward_mean": 13
       }

ModelCatalog.register_custom_model("pa_model",TorchParametricActionsModel)

results = tune.run(
    # SACTrainer,
    PPOTrainer,
    #ImpalaTrainer,
    #A3CTrainer,
     config={
    "env": MultiAgentSoccer,
    "model":{
        "custom_model": "pa_model",
        "custom_model_config":{
            "true_obs_shape":origin_obs_space
            }
        },
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
    "num_workers": 2,
    "output": "/tmp/out-soccer", 
    "log_level": "INFO",
    "batch_mode": "complete_episodes",
    "framework": 'torch'
}, stop=stop)  

# import pickle
# print("best lr:",results.get_best_config(metric="mean_loss",mode="min"))
# fw=open("bestcofig.pkl","wb")
# pickle.dump(results.get_best_config(metric="mean_loss",mode="min"),fw)
# fw.close()