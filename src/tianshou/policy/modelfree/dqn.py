from copy import deepcopy
from typing import Any, Self, cast

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    BatchWithReturnsProtocol,
    ModelOutputBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy import BasePolicy
from tianshou.policy.base import TLearningRateScheduler


class DQNPolicy(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).

    :param model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param optim: a torch.optim for optimizing the model.
    :param discount_factor: in [0, 1].
    :param estimation_step: the number of steps to look ahead.
    :param target_update_freq: the target network update frequency (0 if
        you do not use the target network).
    :param reward_normalization: normalize the **returns** to Normal(0, 1).
        TODO: rename to return_normalization?
    :param is_double: use double dqn.
    :param clip_loss_grad: clip the gradient of the loss in accordance
        with nature14236; this amounts to using the Huber loss instead of
        the MSE loss.
    :param observation_space: Env's observation space.
    :param lr_scheduler: if not None, will be called in `policy.update()`.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        # TODO: type violates Liskov substitution principle
        action_space: gym.spaces.Discrete,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        observation_space: gym.Space | None = None,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=False,
            action_bound_method=None,
            lr_scheduler=lr_scheduler,
        )
        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert (
            0.0 <= discount_factor <= 1.0
        ), f"discount factor should be in [0, 1] but got: {discount_factor}"
        self.gamma = discount_factor
        assert (
            estimation_step > 0
        ), f"estimation_step should be greater than 0 but got: {estimation_step}"
        self.n_step = estimation_step
        self._target = target_update_freq > 0
        self.freq = target_update_freq
        self._iter = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self.rew_norm = reward_normalization
        self.is_double = is_double
        self.clip_loss_grad = clip_loss_grad

        # TODO: set in forward, fix this!
        self.max_action_num: int

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode: bool = True) -> Self:
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        if self.is_double:
            return target_q[np.arange(len(result.act)), result.act]
        # Nature DQN, over estimate
        return target_q.max(dim=1)[0]

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        return self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.n_step,
            rew_norm=self.rew_norm,
        )

    def compute_q_value(self, logits: torch.Tensor, mask: np.ndarray | None) -> torch.Tensor:
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def forward(
        self,
        batch: RolloutBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden = model(obs_next, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        result = Batch(logits=logits, act=act, state=hidden)
        return cast(ModelOutputBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> dict[str, float]:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q

        if self.clip_loss_grad:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def exploration_noise(
        self,
        act: np.ndarray | BatchProtocol,
        batch: RolloutBatchProtocol,
    ) -> np.ndarray | BatchProtocol:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act
