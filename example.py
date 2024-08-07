import gymnasium as gym
import nav2d_env 

if __name__ == "__main__":
    env = gym.make(id="Nav2D-v0", n=3, render_mode="human")
    env.reset()
    for _ in range(300):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        if truncated:
            print("Truncated")
            break

        if terminated:
            print("Terminated")
            break
        env.render()
    env.close()