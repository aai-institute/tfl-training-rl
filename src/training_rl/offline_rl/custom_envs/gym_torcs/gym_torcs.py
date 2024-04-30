import training_rl
from training_rl.offline_rl.load_env_variables import load_env_variables
load_env_variables()

import subprocess
from typing import Any

import gymnasium as gym
from gymnasium import Env

import training_rl.offline_rl.custom_envs.gym_torcs.snakeoil3_gym as snakeoil3
import numpy as np
import copy
import os
import time

from training_rl.offline_rl.config import get_gym_torcs_abs_path

module_path = training_rl.__file__
root_directory = os.path.dirname(module_path)

class TorcsEnv (Env):
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True


    def __init__(self, vision=False, throttle=False, gear_change=False, render_mode = None):
       #print("Init")
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True

        ##print("launch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime  -vision &')
        else:
            os.system('torcs -nofuel -nodamage -nolaptime &')
        time.sleep(0.5)
        autostart_path = os.path.join(get_gym_torcs_abs_path(), "autostart.sh")
        subprocess.run(['sh', autostart_path])
        time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs



        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        if vision is False:
            self.observation_space = gym.spaces.Dict({
                'opponents': gym.spaces.Box(low=np.float32(0.0), high=np.float32(2.0), shape=(36,), dtype=np.float32),
                'rpm': gym.spaces.Box(low=np.float32(0.0), high=np.float32(np.inf), shape=(), dtype=np.float32),
                'focus': gym.spaces.Box(low=np.float32(-1.0), high=np.float32(1.0), shape=(5,), dtype=np.float32),
                'track': gym.spaces.Box(low=np.float32(-2.0), high=np.float32(2.0), shape=(19,), dtype=np.float32),
                'speedX': gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(), dtype=np.float32),
                'speedY': gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(), dtype=np.float32),
                'speedZ': gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(), dtype=np.float32),
                'wheelSpinVel': gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(4,),
                                               dtype=np.float32),
                'trackPos': gym.spaces.Box(low=np.float32(-2.0), high=np.float32(2.0), shape=(), dtype=np.float32),
                'angle': gym.spaces.Box(low=np.float32(-np.pi), high=np.float32(np.pi), shape=(), dtype=np.float32),
                'curLapTime': gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(), dtype=np.float32)
            })
        else:
            raise ValueError("Vision is not implemented in this environment")

    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            """
            if client.S.d['speedX'] > 50:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 80:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 110:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 140:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 170:
                action_torcs['gear'] = 6
            """

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of torcs
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from torcs
        self.observation = self.make_observation(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])
        progress = sp*np.cos(obs['angle'])
        #reward = progress

        # collision detection
        #if obs['damage'] - obs_pre['damage'] > 0:
        #    reward = -1

        # Termination judgement #########################
        episode_terminate = False
        if track.min() < 0:  # Episode is terminated if the car is out of track
            #reward = - 1
            episode_terminate = True
            client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                episode_terminate = True
                client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True

        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1
        reward = self.reward(obs, obs_pre)


    #ToDo: Truncated is added like False: Check how it is defined in gymnasium.
        return self.get_obs(), reward, client.R.d['meta'], False, {}

    def reward(self, car_obs, car_prev_obs):
        sp = np.array(car_obs['speedX'])
        progress = sp * np.cos(car_obs['angle'])/(2.0*self.default_speed)
        reward = progress

        reward -= np.abs(car_obs["trackPos"] )

        # collision detection
        #if car_obs['damage'] - car_prev_obs['damage'] > 0:
        #    reward = -1

        track = np.array(car_obs['track'])

        if track.min() < 0:  # Episode is terminated if the car is out of track
            reward = - 1

        return reward

    def reset(self,
              relaunch=False,
              seed: int | None = None,
              options: dict[str, Any] | None = None,
              ):

        relaunch = False
        #print("Reset")
        super().reset()

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting torcs every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### torcs is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3001, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observation(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs(), {}

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nodamage -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': u[2]})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list 
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observation(self, raw_obs):
        obs = {
                "focus": np.array(raw_obs['focus'], dtype=np.float32)/200.,
                "speedX": np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                "speedY": np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                "speedZ": np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                "opponents": np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                "rpm": np.array(raw_obs['rpm'], dtype=np.float32),
                "track": np.array(raw_obs['track'], dtype=np.float32)/200.,
                "wheelSpinVel": np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                "trackPos": np.array(raw_obs['trackPos'], dtype=np.float32),
                "angle": np.array(raw_obs['angle'], dtype=np.float32),
                'curLapTime': np.array(raw_obs['lastLapTime'], dtype=np.float32)

        }
        return obs


class TorcsLidarEnv(TorcsEnv):
    def __init__(self, vision=False, throttle=False, gear_change=False, render_mode = None):

        super(TorcsLidarEnv, self).__init__(
            vision=vision,
            throttle=throttle,
            gear_change=gear_change,
            render_mode=render_mode
        )
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)

    @staticmethod
    def _preprocess_observation(observation):
        preprocessed_observation = np.array([lidar for lidar in observation["track"]])
        return preprocessed_observation

    def reset(self, **kwargs):
        # Reset the environment and preprocess the initial observation
        observation, info = super().reset(**kwargs)
        self._raw_observation = observation
        preprocessed_observation = self._preprocess_observation(observation)
        return preprocessed_observation, info

    def step(self, action):
        # Take a step in the environment and preprocess the observation
        action = np.clip(action, -0.1, 0.1)
        observation, reward, done, truncations, info = super().step(action)
        self._raw_observation = observation
        preprocessed_observation = self._preprocess_observation(observation)
        return preprocessed_observation, reward, done, truncations, info

    @property
    def raw_observation(self):
        return self._raw_observation

    def number_steps(self):
        return self.time_step
