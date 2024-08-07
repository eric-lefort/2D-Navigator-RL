from nav2d_env import Nav2DEnv
import numpy as np

if __name__ == "__main__":
    render_mode="human"
    save_gif = True
    if save_gif:
        from PIL import Image
        render_mode = "rgb_array"
        frames = []

    env = Nav2DEnv(n=3, render_mode=render_mode)
    env.reset()
    # x, y, vx, vy, goal x, goal y
    observation = np.array([0, 0, 0, 0, 0, 0])
    for _ in range(500):
        x, y, vx, vy, goal_x, goal_y = observation
        distance = np.array([goal_x - x, goal_y - y], dtype=np.float32)
        action_components = env.action_components

        action = np.clip(action_components @ distance, -1, 1, dtype=np.float32)
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
        gif_path = 'assets/example_p_ctrl.gif'
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=20, loop=0)