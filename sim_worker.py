"""
LIBERO Sim Worker — runs episodes on GPU, calls Ray Serve for actions.
"""
import ray
import numpy as np
import time
import os
import builtins

os.environ["MUJOCO_GL"] = "egl"
os.environ["DISPLAY"] = ""
builtins.input = lambda *args, **kwargs: "n"


@ray.remote(num_gpus=1)
class SimWorker:
    """Runs LIBERO episodes on a GPU worker, calls policy server for actions."""

    def __init__(self, task_suite="libero_object", task_idx=0, render_size=256):
        import os, builtins
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["DISPLAY"] = ""
        builtins.input = lambda *args, **kwargs: "n"

        self.task_suite = task_suite
        self.task_idx = task_idx
        self.render_size = render_size
        self.env = None

    def _ensure_env(self):
        if self.env is None:
            from libero_env import LiberoEnv
            self.env = LiberoEnv(
                task_suite=self.task_suite,
                task_idx=self.task_idx,
                render_size=self.render_size,
            )

    def run_episode(self, policy_handle, max_steps=300, record_video=False):
        self._ensure_env()

        obs = self.env.reset()
        language = self.env.get_language_instruction()

        total_reward = 0.0
        frames = []
        latencies = []
        done = False

        t_start = time.time()

        for step in range(max_steps):
            if record_video:
                frames.append(obs["agentview_image"].copy())

            policy_input = {
                "agentview_image": obs["agentview_image"],
                "wrist_image": obs["wrist_image"],
                "robot_state": obs["robot_state"],
                "gripper_state": obs["gripper_state"],
                "language_instruction": language,
            }

            result = policy_handle.predict.remote(policy_input).result()
            action = result["action"]
            latencies.append(result["latency_ms"])

            obs, reward, done, info = self.env.step(action)
            total_reward += reward

            if done:
                if record_video:
                    frames.append(obs["agentview_image"].copy())
                break

        elapsed = time.time() - t_start
        success = info.get("success", False) if isinstance(info, dict) else False

        result = {
            "task": self.env.task_name,
            "language": language,
            "steps": step + 1,
            "total_reward": total_reward,
            "success": success,
            "elapsed_s": elapsed,
            "avg_policy_latency_ms": np.mean(latencies) if latencies else 0,
            "fps": (step + 1) / elapsed,
        }

        if record_video:
            result["frames"] = frames

        return result

    def close(self):
        if self.env is not None:
            self.env.close()


def save_gif(frames, path, fps=15):
    import imageio
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"Saved GIF: {path} ({len(frames)} frames)")
