# Rendering didn't work without
# conda install -c conda-forge libstdcxx-ng

import gym
import cv2
import torch
from scripts_offline_rl.transformer_model import DecisionTransformer, evaluate_on_env

device = "cpu"

path_to_model = "/scripts_offline_rl/model/model.pt"

env_name = "Walker2d-v3"
#env_name = "CartPole-v1"
eval_env = gym.make(env_name, render_mode='rgb_array')

eval_env.reset()

#for i in range(1000):
#    eval_env.step(eval_env.action_space.sample())
#    image = eval_env.render()
#    quit = show_video(image)
#    if quit:
#        break



state_dim = eval_env.observation_space.shape[0]
act_dim = eval_env.action_space.shape[0]
n_blocks = 3            # num of transformer blocks
embed_dim = 128         # embedding (hidden) dim of transformer
n_heads = 1             # num of transformer heads
dropout_p = 0.1         # dropout probability
context_len = 20        # K in decision transformer

eval_rtg_target = 5000
eval_rtg_scale = 1000		# normalize returns to go
num_test_eval_ep = 1000			# num of evaluation episodes
eval_max_eval_ep_len = 10000		# max len of one episode



eval_model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=embed_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=dropout_p,
        ).to(device)


# load checkpoint
eval_model.load_state_dict(torch.load(path_to_model, map_location=device))

# evaluate on env
results = evaluate_on_env(
    eval_model,
    device,
    context_len,
    eval_env,
    eval_rtg_target,
    eval_rtg_scale,
    num_test_eval_ep,
    eval_max_eval_ep_len,
    #eval_state_mean,
    #eval_state_std
    render=True
)
