import pygame
import numpy as np
from typing import Optional, Union
import gymnasium as gym
from gymnasium.envs.registration import register

class Nav2DEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """A simple 2d navigation environment with redundant actions. The agent controls a navigator that moves in a 2D plane. The agent has control over 2^n velocity vectors evenly spaced around the unit circle and moves in the direction of the average of the input vectors.
    
    State:
        x: float
        y: float
        vx: float
        vy: float
        goal_x: float
        goal_y: float

    Observations:
        Type:
            Box(4,)

        goal - x: float
        goal - y: float
        vx: float
        vy: float
    Actions:
        Type: Box(-1, 1, 2^n,)
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, n: int = 1, render_mode: str = "human"):
        """
        Args:
            n: int
                2^n is the number of actions to control velocity vector.
            render_mode: str
                The mode to render the environment in.
        """

        assert n > 0, "number of actions (2^n) must be greater than 1 to make the environment controllable."
        assert render_mode in self.metadata["render_modes"], f"Invalid render mode {render_mode}"
        self.render_mode = render_mode
        self.dense_reward = True
        self.n_max_steps = 250
        self.visualize_actions = True

        self.screen_size = (300, 300)
        self.screen_center = (self.screen_size[0] // 2, self.screen_size[1] // 2)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None
        self.action = None

        # Initialize Pygame
        pygame.init()
        flags = pygame.HIDDEN if render_mode == "rgb_array" else 0
        self.screen = pygame.display.set_mode(self.screen_size, flags)
        self.screen.set_alpha(None)
        self.clock = pygame.time.Clock()
        
        # Define colors if needed
        self.bg_color = (255, 255, 255)  # White background
        self.circle_color = (0, 0, 0) # (50, 225, 30)  # Green
        self.goal_color = (0, 0, 0) # (255, 0, 0)  # Red

        self.x_range = (-10, 10)
        self.y_range = (-10, 10)
        self.vx_range = (-10, 10)
        self.vy_range = (-10, 10)

        self.min_distance = np.linalg.norm(np.array([self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0]])) / 12

        self.target_size = 0.5
        self.noise_std = 0.0

        self.action_space = gym.spaces.Box(-1, 1, (2 ** n,))
        self.action_angles = np.linspace(0, np.pi, 2 ** n, endpoint=False)
        self.action_normalization = np.sum(np.sin(self.action_angles))
        self.action_components = np.array([np.cos(self.action_angles), np.sin(self.action_angles)]).T / self.action_normalization
        observation_range = np.array([
            self.x_range, # x
            self.y_range, # y
            self.vx_range, # vx
            self.vy_range, # vy
            # self.x_range, # x goal
            # self.y_range # y goal
        ])
        self.observation_space = gym.spaces.Box(observation_range[:, 0], observation_range[:, 1], (4,))


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

        self.state[:4] = np.array([x, y, vx, vy])
        self.n_steps += 1
        if self.n_steps >= self.n_max_steps:
            truncated = True

        # observation, reward, terminated, truncated, info
        observation = self._get_observation()
        return observation, reward, terminated, truncated, {}
        
    def _get_observation(self):
        x, y, vx, vy, goal_x, goal_y = self.state
        return np.array([(goal_x - x) / 2, (goal_y - y) / 2, vx, vy], dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        x, y, goal_x, goal_y = 0, 0, 0, 0
        while np.linalg.norm([x - goal_x, y - goal_y]) < self.min_distance:
            x = 0.8 * np.random.uniform(*self.x_range)
            y = 0.8 * np.random.uniform(*self.y_range)
            vx = 0.05 * np.random.uniform(*self.x_range)
            vy = 0.05 * np.random.uniform(*self.y_range)
            goal_x = 0.7 * np.random.uniform(*self.x_range)
            goal_y = 0.7 * np.random.uniform(*self.y_range)

        self.state = np.array([x, y, vx, vy, goal_x, goal_y], dtype=np.float32)
        observation = self._get_observation()
        self.n_steps = 0
        info = {}
        return observation, info

    def render(self):
        if not self.isopen:
            return
        
        self.screen.fill(self.bg_color)

        # Draw circle and X
        circle_size = self.screen_size[0] // 40
        pos = self.coordinate_to_pixel(self.state[:2])
        pygame.draw.circle(self.screen, self.circle_color, (int(pos[0]), int(pos[1])), circle_size)

        if self.visualize_actions and self.action is not None:
            sign = np.sign(self.action)
            magnitude = np.abs(self.action)
            for i in range(len(self.action)):
                angle = self.action_angles[i]
                if sign[i] == -1:
                    angle += np.pi
                pos = self.state[:2]
                color = (255, 255 * (1 - magnitude[i]), 255 * (1 - magnitude[i]))

                self.draw_force(pos, angle, color, r=1.5, scale=0.5)

        goal = self.coordinate_to_pixel(self.state[4:])
        x_length = self.screen_size[0] // 60
        pygame.draw.line(self.screen, self.goal_color, (int(goal[0] - x_length), int(goal[1] - x_length)),
                         (int(goal[0] + x_length), int(goal[1] + x_length)), 2)
        pygame.draw.line(self.screen, self.goal_color, (int(goal[0] - x_length), int(goal[1] + x_length)),
                         (int(goal[0] + x_length), int(goal[1] - x_length)), 2)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
            return
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen)  # Retrieves the pixel data from the surface as a NumPy array.
        else:
            raise ValueError(f"Invalid render mode {self.render_mode}")

    def coordinate_to_pixel(self, pos):
        x_factor = self.screen_size[0] / (self.x_range[1] - self.x_range[0])
        y_factor = self.screen_size[1] / (self.y_range[1] - self.y_range[0])
        return (int(x_factor * pos[0] + self.screen_center[0]), int(-y_factor * pos[1] + self.screen_center[1]))

    def draw_force(self, center, theta, color, r=1, scale=1):
        """Position is center of the circle, we place the force pointing outwards at coordinates (r, theta) in polar coordinates."""

        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                              [np.sin(theta),  np.cos(theta)]])
        base_arrow = np.array([
            [r, 0], 
            [r - scale, scale / 4], 
            [r - scale, -scale / 4]
        ])
        arrow = np.dot(base_arrow, rot_matrix.T) + center
        for i in range(3):
            arrow[i] = self.coordinate_to_pixel(arrow[i])
        
        pygame.draw.polygon(self.screen, color, arrow)

    def close(self):
        self.isopen = False
        pygame.quit()

register(
    id='Nav2D-v0',
    entry_point='nav2d_env:Nav2DEnv',
    max_episode_steps=250
)
