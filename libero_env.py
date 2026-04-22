"""
LIBERO environment wrapper for Ray.
Unlike Isaac Lab, LIBERO (robosuite/MuJoCo) doesn't need subprocess isolation.
It runs cleanly inside Ray worker processes.
"""
import numpy as np
import os
import builtins

os.environ["MUJOCO_GL"] = "egl"
os.environ["DISPLAY"] = ""
builtins.input = lambda *args, **kwargs: "n"


class LiberoEnv:
    """Wraps a LIBERO task for use in Ray workers."""

    def __init__(self, task_name=None, task_suite="libero_object", task_idx=0, render_size=256):
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        self.render_size = render_size
        self.action_dim = 7

        bench = benchmark.get_benchmark(task_suite)()
        task_names = bench.get_task_names()

        if task_name is not None:
            assert task_name in task_names, f"Task '{task_name}' not in {task_suite}. Available: {task_names}"
            task_idx = task_names.index(task_name)

        self.task_name = task_names[task_idx]
        task = bench.get_task(task_idx)
        bddl_path = bench.get_task_bddl_file_path(task_idx)
        self.language_instruction = task.language

        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=render_size,
            camera_widths=render_size,
            camera_names=["agentview", "robot0_eye_in_hand"],
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
        )
        self.env.seed(42)

        print(f"[LiberoEnv] Ready: {self.task_name}")
        print(f"  Language: {self.language_instruction}")
        print(f"  Action dim: {self.action_dim}")

    def reset(self):
        obs = self.env.reset()
        return self._process_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), reward, done, info

    def _process_obs(self, obs):
        return {
            "agentview_image": obs["agentview_image"][::-1].copy(),
            "wrist_image": obs["robot0_eye_in_hand_image"][::-1].copy(),
            "robot_state": obs.get("robot0_eef_pos", np.zeros(3)),
            "gripper_state": obs.get("robot0_gripper_qpos", np.zeros(2)),
        }

    def get_language_instruction(self):
        return self.language_instruction

    def close(self):
        self.env.close()
