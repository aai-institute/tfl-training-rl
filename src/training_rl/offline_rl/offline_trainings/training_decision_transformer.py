# Code adapted from https://colab.research.google.com/github/nikhilbarhate99/min-decision-transformer/
# blob/master/min_decision_transformer.ipynb
import os.path

# ToDo: Improve code quality
from training_rl.offline_rl.load_env_variables import load_env_variables
load_env_variables()

import gym
import torch
import pickle
import numpy as np
import cv2
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from datetime import datetime
from training_rl.offline_rl.offline_policies.decision_transformer_policy import (
    get_decision_transformer_default_config, create_decision_transformer_policy_from_dict
)
from training_rl.offline_rl.offline_policies.minimal_GPT_model import DecisionTransformer
from training_rl.offline_rl.utils import D4RLTrajectoryDataset
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter


REF_MAX_SCORE = {
    'halfcheetah' : 12135.0,
    'walker2d' : 4592.3,
    'hopper' : 3234.3,
}

REF_MIN_SCORE = {
    'halfcheetah' : -280.178953,
    'walker2d' : 1.629008,
    'hopper' : -20.272305,
}


def get_d4rl_normalized_score(score, env_name):
    env_key = env_name.split('-')[0].lower()
    assert env_key in REF_MAX_SCORE, f'no reference score for {env_key} env to calculate d4rl score'
    return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


def get_optimizer_scheduler(
        lr: float = 1e-4,
        wt_decay: float = 1e-4,
        warmup_steps: int = 10000,
) -> (torch.optim.Optimizer, lr_scheduler.LambdaLR):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wt_decay
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    return optimizer, scheduler


def show_video(image, fps=1):
    """
    Show a video from a list of images.

    Args:
        images (list): List of images (numpy arrays).
        fps (int, optional): Frames per second of the video. Defaults to 30.
    """

    cv2.imshow("Video", image)
    key = cv2.waitKey(10)  # Wait for the specified time between frames
    if key & 0xFF == ord('q'):        # Close OpenCV windows
        cv2.destroyAllWindows()
        return True
    return False


def evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000, render=False):

    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    cumulative_reward_per_episode = []
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():

        for _ in range(num_eval_ep):

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                  dtype=torch.float32, device=device)

            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                 dtype=torch.float32, device=device)

            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                        dtype=torch.float32, device=device)

            # init episode
            running_state, _ = env.reset()
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            quit = False

            cumulative_reward_single_episode_list = []
            cumulative_reward_single_episode_value = 0
            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)

                # calculate running rtg and add in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg

                if t < context_len:
                    _, act_preds, _ = model.forward(timesteps[:, :context_len],
                                                    states[:, :context_len],
                                                    actions[:, :context_len],
                                                    rewards_to_go[:, :context_len])
                    act = act_preds[0, t].detach()
                else:
                    _, act_preds, _ = model.forward(timesteps[:, t - context_len + 1:t + 1],
                                                    states[:, t - context_len + 1:t + 1],
                                                    actions[:, t - context_len + 1:t + 1],
                                                    rewards_to_go[:, t - context_len + 1:t + 1])
                    act = act_preds[0, -1].detach()

                running_state, running_reward, done, _, _ = env.step(act.cpu().numpy())

                cumulative_reward_single_episode_value += running_reward
                cumulative_reward_single_episode_list.append(cumulative_reward_single_episode_value)

                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward

                if render:
                    image = env.render()
                    quit = show_video(image)
                    if quit:
                        break

                if done:
                    break

            if quit:
                break
            cumulative_reward_per_episode.append(cumulative_reward_single_episode_list)

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep
    results['eval/cumulative_reward_per_episode'] = cumulative_reward_per_episode

    return results


def train_decision_transformer(
        traj_data_loader: DataLoader,
        num_epochs: int,
        num_steps_per_epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        model: DecisionTransformer,
        env: gym.Env,
        context_len: int,
        device:str = "cuda",
):

    log_dir = PATH_TO_MODELS

    data_iter = iter(traj_data_loader)
    total_updates = 0
    start_time = datetime.now().replace(microsecond=0)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    for i_train_iter in range(num_epochs):

        log_action_losses = []
        model.train()

        cum_reward = 0.0
        for _ in range(num_steps_per_epoch):
            try:
                time_steps, states, actions, returns_to_go, traj_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(traj_data_loader)
                time_steps, states, actions, returns_to_go, traj_mask = next(data_iter)

            time_steps = time_steps.to(device)  # B x T
            states = states.to(device)  # B x T x state_dim
            actions = actions.to(device)  # B x T x act_dim
            returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1)  # B x T x 1
            traj_mask = traj_mask.to(device)  # B x T

            action_target = torch.clone(actions).detach().to(device)

            state_preds, action_preds, return_preds = model.forward(
                timesteps=time_steps,
                states=states,
                actions=actions,
                returns_to_go=returns_to_go
            )

            cum_reward += torch.mean(torch.mean(return_preds, dim=1))

            # only consider non padded elements
            act_dim = env.action_space.shape[0]
            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1, ) > 0]
            action_target = action_target.view(-1, act_dim)[traj_mask.view(-1, ) > 0]

            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            log_action_losses.append(action_loss.detach().cpu().item())

        # evaluate on env
        results = evaluate_on_env(model, device, context_len, env, RTG_TARGET, RTG_SCALE, NUM_EVAL_EP, MAX_EVAL_EP_LEN)
        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']

        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_steps_per_epoch

        log_str = (
                "=" * 60 + '\n' + "time elapsed: " + time_elapsed + '\n' + "num of updates: " + str(total_updates) +
                '\n' + "action loss: " + format(mean_action_loss, ".5f") + '\n' + "eval avg reward: " +
                format(eval_avg_reward, ".5f") + '\n' + "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n'
        )

        print(log_str)

        # Log metrics to TensorBoard
        writer.add_scalar('Cumulative Reward', cum_reward, total_updates)
        writer.add_scalar('Action Loss', mean_action_loss, total_updates)
        writer.add_scalar('Eval Avg Reward', eval_avg_reward, total_updates)
        writer.add_scalar('Eval Avg Ep Len', eval_avg_ep_len, total_updates)

        print("saving current model at: " + PATH_TO_MODEL)
        torch.save(model.state_dict(), PATH_TO_MODEL)

    end_time = datetime.now().replace(microsecond=0)
    total_time_minutes = (end_time - start_time).total_seconds() / 60.0
    print("Total time taken (minutes):", total_time_minutes)


if __name__ == "__main__":

    PATH_TO_MODELS = "../data/decision_transformers/models/"

    #DATASET_PATH = "../data/decision_transformers/d4rl_data/halfcheetah-medium-v0.pkl"
    #PATH_TO_MODEL = os.path.join(PATH_TO_MODELS, "model_d4rl_halfcheetah_medium_v0_April_24_v2.pt")
    #ENV_NAME = "HalfCheetah-v3"

    DATASET_PATH = "../data/decision_transformers/d4rl_data/walker2d-medium-v1.pkl"
    PATH_TO_MODEL = os.path.join(PATH_TO_MODELS, "model_d4rl_walker2d-medium-v1_April_24_v0.pt")
    ENV_NAME = "Walker2d-v3"

    BATCH_SIZE = 64
    NUM_EPOCHS = 2000
    NUM_STEPS_PER_EPOCH = 1000

    RTG_TARGET = 3500  # only for evaluation
    RTG_SCALE = 1000  # scale to normalize returns to go

    # Evaluation during training
    MAX_EVAL_EP_LEN = 1000  # max len of one evaluation episode
    NUM_EVAL_EP = 1  # num of evaluation episodes per iteration

    RESTORE_TRAINING = False

    env = gym.make(ENV_NAME)
    decision_transformer_config = get_decision_transformer_default_config()
    decision_transformer_config["device"] = "cuda"
    device = decision_transformer_config["device"]
    context_len = decision_transformer_config["context_len"]

    model = create_decision_transformer_policy_from_dict(
        config=decision_transformer_config,
        action_space=env.action_space,
        observation_space=env.observation_space
    )

    if RESTORE_TRAINING:
        model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=device))
        print("Policy loaded from: ", PATH_TO_MODEL)

    optimizer, scheduler = get_optimizer_scheduler()

    with open(DATASET_PATH, 'rb') as f:
        trajectories = pickle.load(f)
    traj_dataset = D4RLTrajectoryDataset(
        DATASET_PATH,
        context_len,
        RTG_SCALE)

    traj_data_loader = DataLoader(
        dataset=traj_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    train_decision_transformer(
        traj_data_loader=traj_data_loader,
        num_epochs=NUM_EPOCHS,
        num_steps_per_epoch=NUM_STEPS_PER_EPOCH,
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        env=env,
        context_len=context_len,
        device=device,
    )
