from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch, to_torch_as
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    BatchWithReturnsProtocol,
    ModelOutputBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy import DQNPolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.utils.net.common import BranchingNet


class BranchingDQNPolicy(DQNPolicy):
    """Implementation of the Branching dual Q network arXiv:1711.08946.

    :param model: BranchingNet mapping (obs, state, info) -> logits.
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
        model: BranchingNet,
        optim: torch.optim.Optimizer,
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
        assert (
            estimation_step == 1
        ), f"N-step bigger than one is not supported by BDQ but got: {estimation_step}"
        super().__init__(
            model=model,
            optim=optim,
            action_space=action_space,
            discount_factor=discount_factor,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
            is_double=is_double,
            clip_loss_grad=clip_loss_grad,
            observation_space=observation_space,
            lr_scheduler=lr_scheduler,
        )
        self.model = cast(BranchingNet, self.model)

    # TODO: mypy complains b/c max_action_num is declared in base class, see todo there
    @property
    def max_action_num(self) -> int:  # type: ignore
        return self.model.action_per_branch  # type: ignore

    @property
    def num_branches(self) -> int:
        return self.model.num_branches  # type: ignore

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        if self.is_double:
            act = np.expand_dims(self(batch, input="obs_next").act, -1)
            act = to_torch(act, dtype=torch.long, device=target_q.device)
        else:
            act = target_q.max(-1).indices.unsqueeze(-1)
        return torch.gather(target_q, -1, act).squeeze()

    def _compute_return(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        gamma: float = 0.99,
    ) -> BatchWithReturnsProtocol:
        rew = batch.rew
        with torch.no_grad():
            target_q_torch = self._target_q(buffer, indice)  # (bsz, ?)
        target_q = to_numpy(target_q_torch)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        end_flag = end_flag[indice]
        mean_target_q = np.mean(target_q, -1) if len(target_q.shape) > 1 else target_q
        _target_q = rew + gamma * mean_target_q * (1 - end_flag)
        target_q = np.repeat(_target_q[..., None], self.num_branches, axis=-1)
        target_q = np.repeat(target_q[..., None], self.max_action_num, axis=-1)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return cast(BatchWithReturnsProtocol, batch)

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        """Compute the 1-step return for BDQ targets."""
        return self._compute_return(batch, buffer, indices)

    def forward(
        self,
        batch: RolloutBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden = model(obs_next, state=state, info=batch.info)
        act = to_numpy(logits.max(dim=-1)[1])
        result = Batch(logits=logits, act=act, state=hidden)
        return cast(ModelOutputBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> dict[str, float]:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        act = to_torch(batch.act, dtype=torch.long, device=batch.returns.device)
        q = self(batch).logits
        act_mask = torch.zeros_like(q)
        act_mask = act_mask.scatter_(-1, act.unsqueeze(-1), 1)
        act_q = q * act_mask
        returns = batch.returns
        returns = returns * act_mask
        td_error = returns - act_q
        loss = (td_error.pow(2).sum(-1).mean(-1) * weight).mean()
        batch.weight = td_error.sum(-1).sum(-1)  # prio-buffer
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
            rand_act = np.random.randint(low=0, high=self.max_action_num, size=(bsz, act.shape[-1]))
            if hasattr(batch.obs, "mask"):
                rand_act += batch.obs.mask
            act[rand_mask] = rand_act[rand_mask]
        return act
