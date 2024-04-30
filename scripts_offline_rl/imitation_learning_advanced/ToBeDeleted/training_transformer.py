from training_rl.offline_rl.load_env_variables import load_env_variables
load_env_variables()


import pickle
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from d4rl_torch_dataset_loader import D4RLTrajectoryDataset
from transformer_model import (DecisionTransformer, evaluate_on_env)

#dataset_path = "/home/ivan/Documents/GIT_PROJECTS/tfl-training-rl/scripts_offline_rl/data/walker2d-medium-v1.pkl"
dataset_path = "../../../src/training_rl/offline_rl/data/decision_transformers/d4rl_data/halfcheetah-medium-v0.pkl"
#env_name = "Walker2d-v3"
env_name = "HalfCheetah-v3"

rtg_target = 6000
rtg_scale = 1000        # scale to normalize returns to go
context_len = 20        # K in decision transformer
batch_size = 64

n_blocks = 3            # num of transformer blocks
embed_dim = 128         # embedding (hidden) dim of transformer
n_heads = 1             # num of transformer heads
dropout_p = 0.1         # dropout probability

lr = 1e-4                   # learning rate
wt_decay = 1e-4             # weight decay
warmup_steps = 10000        # warmup steps for lr scheduler

# total updates = max_train_iters x num_updates_per_iter
max_train_iters = 200
num_updates_per_iter = 100

device = "cuda"

save_model_path = "/scripts_offline_rl/model/model_d4rl_halfcheetah_medium_v0_20000iters.pt"


# Evaluation during training
max_eval_ep_len = 1000      # max len of one evaluation episode
num_eval_ep = 10            # num of evaluation episodes per iteration
# used for input normalization
with open(dataset_path, 'rb') as f:
    trajectories = pickle.load(f)

min_len = 10**4
states = []
for traj in trajectories:
    min_len = min(min_len, traj['observations'].shape[0])
    states.append(traj['observations'])
states = np.concatenate(states, axis=0)
state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6


traj_dataset = D4RLTrajectoryDataset(dataset_path, context_len, rtg_scale)

traj_data_loader = DataLoader(traj_dataset,
						batch_size=batch_size,
						shuffle=True,
						pin_memory=True,
						drop_last=True)


#print(traj_dataset.get_state_stats())

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]


model = DecisionTransformer(
			state_dim=state_dim,
			act_dim=act_dim,
			n_blocks=n_blocks,
			h_dim=embed_dim,
			context_len=context_len,
			n_heads=n_heads,
			drop_p=dropout_p,
		).to(device)


optimizer = torch.optim.AdamW(
					model.parameters(),
					lr=lr,
					weight_decay=wt_decay
				)

scheduler = torch.optim.lr_scheduler.LambdaLR(
		optimizer,
		lambda steps: min((steps+1)/warmup_steps, 1)
	)


data_iter = iter(traj_data_loader)

timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

#returns_to_go = returns_to_go.unsqueeze(-1)


#bla = model.forward(
#	timesteps=timesteps,
#	states=states,
#	actions=actions,
#	returns_to_go=returns_to_go
#)


max_d4rl_score = -1.0
total_updates = 0
start_time = datetime.now().replace(microsecond=0)

for i_train_iter in range(max_train_iters):
	log_action_losses = []

	model.train()

	for _ in range(num_updates_per_iter):
		try:
			timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
		except StopIteration:
			data_iter = iter(traj_data_loader)
			timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

		timesteps = timesteps.to(device)  # B x T
		states = states.to(device)  # B x T x state_dim
		actions = actions.to(device)  # B x T x act_dim
		returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1)  # B x T x 1
		traj_mask = traj_mask.to(device)  # B x T

		action_target = torch.clone(actions).detach().to(device)

		state_preds, action_preds, return_preds = model.forward(
			timesteps=timesteps,
			states=states,
			actions=actions,
			returns_to_go=returns_to_go
		)

		# only consider non padded elements
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
	results = evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
							  num_eval_ep, max_eval_ep_len)
	eval_avg_reward = results['eval/avg_reward']
	eval_avg_ep_len = results['eval/avg_ep_len']
	#eval_d4rl_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100

	mean_action_loss = np.mean(log_action_losses)
	time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

	total_updates += num_updates_per_iter


	log_str = ("=" * 60 + '\n' +
			   "time elapsed: " + time_elapsed + '\n' +
			   "num of updates: " + str(total_updates) + '\n' +
			   "action loss: " + format(mean_action_loss, ".5f") + '\n' +
			   "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
			   "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n'
			   #"eval d4rl score: " + format(eval_d4rl_score, ".5f")
			   )

	print(log_str)

	log_data = [time_elapsed, total_updates, mean_action_loss,
				eval_avg_reward, eval_avg_ep_len,
				#eval_d4rl_score
				]

	'''
	csv_writer.writerow(log_data)

	# save model
	print("max d4rl score: " + format(max_d4rl_score, ".5f"))
	if eval_d4rl_score >= max_d4rl_score:
		print("saving max d4rl score model at: " + save_best_model_path)
		torch.save(model.state_dict(), save_best_model_path)
		max_d4rl_score = eval_d4rl_score

	'''
	print("saving current model at: " + save_model_path)
	torch.save(model.state_dict(), save_model_path)


'''
print("=" * 60)
print("finished training!")
print("=" * 60)
end_time = datetime.now().replace(microsecond=0)
time_elapsed = str(end_time - start_time)
end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
print("started training at: " + start_time_str)
print("finished training at: " + end_time_str)
print("total training time: " + time_elapsed)
print("max d4rl score: " + format(max_d4rl_score, ".5f"))
print("saved max d4rl score model at: " + save_best_model_path)
print("saved last updated model at: " + save_model_path)
print("=" * 60)

csv_writer.close()
'''



