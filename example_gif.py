import numpy as np
from PIL import Image
from nav2d_env import Nav2DEnv

if __name__ == "__main__":
    env = Nav2DEnv(n=3, render_mode="rgb_array")
    env.reset()

    frames = []
    env.state[:2] = np.array([0, 0])
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, _, _ = env.step(action)
        frame = env.render()
        binary_frame = np.mean(frame, axis=-1)
        frames.append(binary_frame)
        if terminated:
            print("Terminated")
            break
    
    # Convert frames to GIF
    gif_path = 'assets/example_rand_act.gif'
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=20, loop=0)

    env.close()