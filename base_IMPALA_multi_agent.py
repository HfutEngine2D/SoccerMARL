#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ray import tune
from ray.rllib.agents.a3c import A3CTrainer
from ray.tune import grid_search
from gym import spaces  #space include observation_space and action_space
import numpy as np
import hfo_py
import torch
import argparse
from soccer_env.mult_agent_env import MultiAgentSoccer

parser = argparse.ArgumentParser()

parser.add_argument("--run", type=str, default="impala")
parser.add_argument("--restore", type=str, default=None)
parser.add_argument("--local-dir", type=str, default="~/ray_results")
# parser.add_argument("--as-test", action="store_true")
parser.add_argument("--defense-npcs", type=int, default=1)
parser.add_argument("--num-workers", type=int, default=1)
parser.add_argument("--offline", type=float, default=0.0)
parser.add_argument("--offense-agents", type=int, default=2)
parser.add_argument("--stop-iters", type=int, default=800)
parser.add_argument("--stop-timesteps", type=int, default=5000000)
parser.add_argument("--stop-reward", type=float, default=100.0)

if __name__ == "__main__":
    args = parser.parse_args()
    
    inputdir = "/run/media/caprlith/data/"+str(args.offense_agents)+"v"+str(args.defense_npcs)+"/"
    if args.run == "impala":
        from ray.rllib.agents.impala import ImpalaTrainer
    else:
        from agents.impala_klloss import ImpalaTrainer

    env_config = {
        "server_config": {
            "defense_npcs": args.defense_npcs,
            "offense_agents": args.offense_agents
        },
        " feature_set": hfo_py.LOW_LEVEL_FEATURE_SET,
    }


    def on_episode_end(info):
        episode = info["episode"]
        # print("episode.last_info_for()", episode.last_info_for(0))
        episode.custom_metrics["goal_rate"] = int(
            episode.last_info_for(0)['status'] == hfo_py.GOAL)

    server_config = env_config["server_config"]
    obs_space_size = 59 + 9 * (
        server_config["defense_npcs"] + server_config["offense_agents"] - 1)
    observation_space = spaces.Box(low=-1, high=1,
                                                shape=((obs_space_size,)), dtype=np.float32)

    act_space = spaces.Discrete(14)

    def gen_policy(_):
        return (None, observation_space, act_space, {})

    # Setup PPO with an ensemble of `num_policies` different policies
    policies = {
        'policy_{}'.format(i): gen_policy(i) for i in range(server_config["offense_agents"]) 
    }
    policy_ids = list(policies.keys())

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward
        }

    # ModelCatalog.register_custom_model("pa_model",TorchParametricActionsModel)
    config={
        "env": MultiAgentSoccer,
        "model":{
            "fcnet_hiddens":[512, 512],
            "fcnet_activation": "relu",
        },
        "env_config": env_config,
        'multiagent': {
            'policies': policies,
            'policy_mapping_fn': lambda agent_id: policy_ids[agent_id],
            "count_steps_by": "agent_steps"
        },
        # "model": {
        #     "fcnet_hiddens":
        #     grid_search([[256, 256], [512, 512], [512, 256, 128],
        #                  [1024, 512, 256], [2048, 1024, 512, 256, 128]]),
        #     "fcnet_activation":
        #     grid_search(["swish", "relu", "tanh"]),
        #     "use_lstm":
        #     grid_search([False, True])
        # },
        "callbacks": {
            "on_episode_end": on_episode_end,
        },
        "lr": 0.0006,
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "num_workers": args.num_workers,
        "log_level":'INFO',
        "framework": 'torch'
    },
    config = config[0]
    if args.run == "bc":
        config.update({
            "input": inputdir,
            "evaluation_num_workers": 1,
            "evaluation_interval": 1,
            "input_evaluation": [],
            "postprocess_inputs": True,
            "evaluation_config":{
                "input": "sampler"}
        })
    if args.offline >0.0:
        config.update({
            "input": {
                "sampler":1-args.offline,
                inputdir: args.offline
            }
        })
    print(config)

        
    results = tune.run(
        ImpalaTrainer,
        # PPOTrainer,
        config = config,
        checkpoint_freq=200,
        checkpoint_at_end=True,
        restore=args.restore,
        local_dir=args.local_dir,
        stop=stop)
