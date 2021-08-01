import logging

import ray
from ray.rllib.agents.impala.vtrace_tf_policy import VTraceTFPolicy
from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution.learner_thread import LearnerThread
from ray.rllib.execution.multi_gpu_learner import TFMultiGPULearner
from ray.rllib.execution.tree_agg import gather_experiences_tree_aggregation
from ray.rllib.execution.common import STEPS_TRAINED_COUNTER, \
    _get_global_vars, _get_shared_metrics
from ray.rllib.execution.replay_ops import MixInReplay
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.concurrency_ops import Concurrently, Enqueue, Dequeue
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.utils.annotations import override
from ray.tune.trainable import Trainable
from ray.tune.resources import Resources
from ray.rllib.agents.impala.impala import DEFAULT_CONFIG, OverrideDefaultResourceRequest, validate_config, execution_plan

logger = logging.getLogger(__name__)


def get_policy_class(config):
    if config["framework"] == "torch":
        if config["vtrace"]:
            from agents.impala_klloss import \
                VTraceTorchPolicy
            return VTraceTorchPolicy
        else:
            from ray.rllib.agents.a3c.a3c_torch_policy import \
                A3CTorchPolicy
            return A3CTorchPolicy
    else:
        if config["vtrace"]:
            return VTraceTFPolicy
        else:
            from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
            return A3CTFPolicy


ImpalaTrainer = build_trainer(
    name="IMPALA",
    default_config=DEFAULT_CONFIG,
    default_policy=VTraceTFPolicy,
    validate_config=validate_config,
    get_policy_class=get_policy_class,
    execution_plan=execution_plan,
    mixins=[OverrideDefaultResourceRequest])
