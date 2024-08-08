import gymnasium as gym
import nav2d_env 
import numpy as np
from gymnasium.envs.registration import register
import torch
from torch.distributions import Normal
from torch import nn

class Nav2DEnvCircular(nav2d_env.Nav2DEnv):
    def __init__(self, n=3, render_mode="human"):
        super().__init__(n, render_mode)
        self.n_max_steps = 5_000

    def step(self, action):
        assert self.action_space.contains(action), f"Action {action} is invalid."
        mass = 2

        self.action = action
        acc = np.dot(action, self.action_components) / mass
        acc += np.random.normal(0, self.noise_std, acc.shape)
        x, y, vx, vy, goal_x, goal_y = self.state
        acc_x, acc_y = acc

        dt = 0.1
        vx += acc_x * dt
        vy += acc_y * dt
        vx = np.clip(vx, *self.vx_range)
        vy = np.clip(vy, *self.vy_range)
        x += vx * dt
        y += vy * dt

        offscreen = bool(
            x < self.x_range[0] 
            or x > self.x_range[1] 
            or y < self.y_range[0] 
            or y > self.y_range[1]
        )
        
        terminated = False
        truncated = False

        reward = 0

        if offscreen:
            reward = -10
            terminated = True

        distance = np.linalg.norm([x - goal_x, y - goal_y])
        if self.dense_reward:
            reward += np.exp(-(distance/3)**2) * 10 / 200
            reward -= np.mean((action)**2 / self.action_normalization) * 0.02
            reward -= 0.005
            if distance < self.target_size:
                terminated = True
                reward += 10
        else:
            if distance < self.target_size:
                terminated = True 
                reward += 1

        target_move_freq = 2
        revolution_len = 200
        r = 5
        if not self.n_steps % target_move_freq:
            i = self.n_steps // target_move_freq
            goal_x = r * np.sin(i * 2* np.pi / revolution_len)
            goal_y = -r * np.cos(i * 2* np.pi / revolution_len)

        self.state = np.array([x, y, vx, vy, goal_x, goal_y], dtype=np.float32)
        self.n_steps += 1
        print(self.n_steps, "/", self.n_max_steps)
        if self.n_steps >= self.n_max_steps:
            truncated = True

        # observation, reward, terminated, truncated, info
        observation = self._get_observation()
        return observation, reward, terminated, truncated, {}

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(env.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

if __name__ == "__main__":
    render_mode="human"
    save_gif = False
    if save_gif:
        from PIL import Image
        render_mode = "rgb_array"
        frames = []

    env = Nav2DEnvCircular(n=3, render_mode=render_mode)
    env.reset()

    model_path = 'runs/Nav2D-v0__ppo_n_3_with_noise__11__1723137314/ppo_n_3_with_noise.cleanrl_model'
    device = 'cpu'
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # x, y, vx, vy, goal x, goal y
    observation = np.array([0, 0, 0, 0])
    for _ in range(2000):
        # with torch.no_grad():
        #     obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        #     action, _ = agent.act(obs_tensor)


        action, _, _, _ = agent.get_action_and_value(torch.Tensor(observation).to(device))
        
        observation, reward, terminated, truncated, _ = env.step(action)
        if truncated:
            print("Truncated")
            break

        if terminated:
            print("Terminated")
            break
        frame = env.render()
        if save_gif:
            binary_frame = np.mean(frame, axis=2)
            frames.append(binary_frame)

    env.close()
    if save_gif:
        gif_path = 'assets/example_pd_ctrl.gif'
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=20, loop=0)