import gym
import logging
import numpy as np

import ray
import ray.rllib.agents.impala.vtrace_torch as vtrace
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    explained_variance, global_norm, sequence_mask

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

class fakeVTraceLoss:

    def __init__(self,pi_loss, vf_loss, entropy):
        self.pi_loss = pi_loss
        self.vf_loss = vf_loss
        self.total_loss = vf_loss + pi_loss
        self.entropy = entropy
        self.value_targets = torch.zeros_like(vf_loss)


def marwil_loss(policy, model, dist_class, train_batch):
    model_out, _ = model.from_batch(train_batch)
    action_dist = dist_class(model_out, model)
    state_values = model.value_function()
    # advantages = train_batch[Postprocessing.ADVANTAGES]
    advantages = 0
    actions = train_batch[SampleBatch.ACTIONS]

    # Advantage estimation.
    adv = advantages - state_values
    adv_squared = torch.mean(torch.pow(adv, 2.0))

    # Value loss.
    policy.v_loss = 0.5 * adv_squared

    # Policy loss.
    # Update averaged advantage norm.
    policy.ma_adv_norm.add_(1e-6 * (adv_squared - policy.ma_adv_norm))
    # Exponentially weighted advantages.
    exp_advs = torch.exp(0 *
                         (adv / (1e-8 + torch.pow(policy.ma_adv_norm, 0.5))))
    # log\pi_\theta(a|s)
    logprobs = action_dist.logp(actions)
    policy.p_loss = -1.0 * torch.mean(exp_advs.detach() * logprobs)

    # Combine both losses.
    policy.total_loss = policy.p_loss + 1.0 * \
        policy.v_loss

    policy.loss = fakeVTraceLoss(policy.p_loss, policy.v_loss, action_dist.entropy())
    # explained_var = explained_variance(advantages, state_values)
    # policy.explained_variance = torch.mean(explained_var)

    return policy.loss.total_loss



def make_time_major(policy, seq_lens, tensor, drop_last=False):
    """Swaps batch and trajectory axis.

    Args:
        policy: Policy reference
        seq_lens: Sequence lengths if recurrent or None
        tensor: A tensor or list of tensors to reshape.
        drop_last: A bool indicating whether to drop the last
        trajectory item.

    Returns:
        res: A tensor with swapped axes or a list of tensors with
        swapped axes.
    """
    if isinstance(tensor, (list, tuple)):
        return [
            make_time_major(policy, seq_lens, t, drop_last) for t in tensor
        ]

    if policy.is_recurrent():
        B = seq_lens.shape[0]
        T = tensor.shape[0] // B
    else:
        # Important: chop the tensor into batches at known episode cut
        # boundaries.
        T = policy.config["rollout_fragment_length"]
        B = tensor.shape[0] // T
    rs = torch.reshape(tensor, [B, T] + list(tensor.shape[1:]))

    # Swap B and T axes.
    res = torch.transpose(rs, 1, 0)

    if drop_last and len(res) > 1:
        return res[:-1]
    return res


def stats(policy, train_batch):
    values_batched = make_time_major(
        policy,
        train_batch.get("seq_lens"),
        policy.model.value_function(),
        drop_last=policy.config["vtrace"])

    return {
        "cur_lr": policy.cur_lr,
        "policy_loss": policy.loss.pi_loss,
        "entropy": policy.loss.entropy,
        "entropy_coeff": policy.entropy_coeff,
        "var_gnorm": global_norm(policy.model.trainable_variables()),
        "vf_loss": policy.loss.vf_loss,
        "vf_explained_var": explained_variance(
            torch.reshape(policy.loss.value_targets, [-1]),
            torch.reshape(values_batched, [-1])),
    }


def choose_optimizer(policy, config):
    if policy.config["opt_type"] == "adam":
        return torch.optim.Adam(
            params=policy.model.parameters(), lr=policy.cur_lr)
    else:
        return torch.optim.RMSprop(
            params=policy.model.parameters(),
            lr=policy.cur_lr,
            weight_decay=config["decay"],
            momentum=config["momentum"],
            eps=config["epsilon"])


def setup_mixins(policy, obs_space, action_space, config):
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    # Create a var.
    policy.ma_adv_norm = torch.tensor(
        [100.0], dtype=torch.float32, requires_grad=False)


VTraceTorchPolicy = build_policy_class(
    name="VTraceTorchPolicy",
    framework="torch",
    loss_fn=marwil_loss,
    get_default_config=lambda: ray.rllib.agents.impala.impala.DEFAULT_CONFIG,
    stats_fn=stats,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=choose_optimizer,
    before_init=setup_mixins,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule],
    get_batch_divisibility_req=lambda p: p.config["rollout_fragment_length"])
