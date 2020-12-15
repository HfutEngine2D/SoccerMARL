#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import grid_search
from gym import spaces
import numpy as np
import hfo_py

from mult_agent_env_demo import MultiAgentSoccer

env_config = {
        "server_config":{
            "defense_npcs": 0,
            "offense_agents":1
        },
        " feature_set": hfo_py.LOW_LEVEL_FEATURE_SET ,
    }
server_config = env_config["server_config"]
obs_space_size = 59 + 9*(server_config["defense_npcs"]+server_config["offense_agents"]-1)
obs_space = spaces.Box(low=-1, high=1,
                                            shape=((obs_space_size,)), dtype=np.float32)
act_space = spaces.Discrete(4)

def gen_policy(_):
    return (None, obs_space, act_space, {})

# Setup PPO with an ensemble of `num_policies` different policies
policies = {
    'policy_{}'.format(i): gen_policy(i) for i in range(1)
}
policy_ids = list(policies.keys())
print("policy_ids", policy_ids)
# print("policies[0][6:]", int(policy_ids[0][6:]))
# print("policy_ids[0", policy_ids[int(policy_ids[0][6:])])

# def outagentids(agent_id):
#     print("--***------*****_____((*****______-----****out agentid", agent_id)
#     return policy_ids[int(agent_id[6:])]

stop = {
       "timesteps_total": 100000,
       "episode_reward_mean": 0.89
       }
results = tune.run(PPOTrainer, config={
    "env": MultiAgentSoccer,
    "env_config": env_config,
    'multiagent': {
        'policies': policies,
        'policy_mapping_fn': tune.function(
            lambda agent_id: policy_ids[agent_id]),
    },
    "lr": 0.001,
    "num_gpus" : 0,
    "num_workers": 1,
    "framework": 'torch'
}, stop=stop)  