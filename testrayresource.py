#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 测试资源分配的脚本，在ray1.2.0以上改用了cgroups
# 会出现死锁，目前我们没能解决。此文件是最小bug示例。
import time

import gym
import numpy as np
import ray
import torch
from gym import error, spaces, utils
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.dqn import DQNTrainer, ApexTrainer

@ray.remote
class SingleAgentEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        self.config = config
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=((3,)), dtype=np.float32)
        self.action_space = spaces.Discrete(14)

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

    @ray.method(num_returns=4)
    def step(self, action):
        print("normal step")
        return np.zeros(3, dtype=np.float32) , 0, False, {'status': 0}

    def reset(self):
        return np.zeros(3, dtype=np.float32)

def make_multiagent(env_name_or_creator):

    class MultiEnv(MultiAgentEnv):
        def __init__(self, config):
            self.num = config["server_config"]["offense_agents"]

            self.agents = []
            for i in range(self.num):
                self.agents.append(env_name_or_creator.remote(config))
                time.sleep(2)
            self.dones = set()

        def reset(self):
            self.dones = set()
            returned = {i: ray.get(stats_id) for i, stats_id in enumerate([a.reset.remote() for a in self.agents])}
            # returned = {i: stats_id for i, stats_id in enumerate([a.reset() for a in self.agents])}
            return returned

        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            # setps = [self.agents[i].step(action) for i, action in action_dict.items()]
            setps = [self.agents[i].step.remote(action) for i, action in action_dict.items()]
            for i, _step in enumerate(setps):
                # obs[i], rew[i], done[i], info[i] = _step
                obs[i], rew[i], done[i], info[i] = ray.get(_step)
                if done[i]:
                    self.dones.add(i)
            done["__all__"] = len(self.dones) == len(self.agents)
            return obs, rew, done, info

    return MultiEnv

MultiAgent = make_multiagent(SingleAgentEnv)


env_config = {
    "server_config": {
        "offense_agents": 1
    }
}

server_config = env_config["server_config"]
obs_space = spaces.Box(
    low=-1, high=1, shape=((3, )), dtype=np.float32)
act_space = spaces.Discrete(14)

def gen_policy(_):
    return (None, obs_space, act_space, {})

# Setup PPO with an ensemble of `num_policies` different policies
policies = {
    'policy_{}'.format(i): gen_policy(i)
    for i in range(server_config["offense_agents"])
}
policy_ids = list(policies.keys())

stop = {"timesteps_total": 20000000, "episode_reward_mean": 10}

results = tune.run(
    ApexTrainer,
    config={
        "env": MultiAgent,
        "env_config": env_config,
        'multiagent': {
            'policies': policies,
            'policy_mapping_fn': lambda agent_id: policy_ids[agent_id],
        },
        "num_gpus": 0,
        "num_workers": 1,
        "framework": 'torch'
    },
    stop=stop)