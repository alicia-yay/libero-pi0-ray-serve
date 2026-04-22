"""Quick test: verify LIBERO runs on a GPU worker with random actions."""
import ray
import numpy as np
import time
import os
import builtins

os.environ["MUJOCO_GL"] = "egl"
os.environ["DISPLAY"] = ""
builtins.input = lambda *args, **kwargs: "n"


@ray.remote(num_gpus=1)
def test_libero_env():
    import os, builtins, numpy as np, time
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["DISPLAY"] = ""
    builtins.input = lambda *args, **kwargs: "n"

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark("libero_object")()
    task = bench.get_task(0)
    bddl_path = bench.get_task_bddl_file_path(0)

    print(f"Task: {task.name}")
    print(f"Language: {task.language}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview", "robot0_eye_in_hand"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )
    env.seed(42)
    obs = env.reset()

    print(f"Agentview image: {obs['agentview_image'].shape}")
    print(f"Wrist image: {obs['robot0_eye_in_hand_image'].shape}")
    print(f"Obs keys: {list(obs.keys())}")

    frames = [obs["agentview_image"][::-1].copy()]
    total_reward = 0

    t0 = time.time()
    for step in range(50):
        action = np.random.uniform(-0.1, 0.1, size=(7,))
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if step % 5 == 0:
            frames.append(obs["agentview_image"][::-1].copy())
        if done:
            break

    elapsed = time.time() - t0
    env.close()

    return {
        "task": task.name,
        "language": task.language,
        "steps": step + 1,
        "total_reward": total_reward,
        "elapsed_s": elapsed,
        "fps": (step + 1) / elapsed,
        "image_shape": list(obs["agentview_image"].shape),
        "num_frames": len(frames),
        "frames": frames,
    }


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    print("=" * 60)
    print("Testing LIBERO on GPU worker...")
    print("=" * 60)

    t0 = time.time()
    result = ray.get(test_libero_env.remote())
    total_time = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Task: {result['task']}")
    print(f"  Language: {result['language']}")
    print(f"  Steps: {result['steps']}")
    print(f"  Total reward: {result['total_reward']:.4f}")
    print(f"  Sim FPS: {result['fps']:.1f}")
    print(f"  Image shape: {result['image_shape']}")
    print(f"  Total time (incl. boot): {total_time:.1f}s")
    print(f"{'=' * 60}")

    import imageio
    gif_path = "/mnt/cluster_storage/libero_test.gif"
    imageio.mimsave(gif_path, result["frames"], fps=5, loop=0)
    print(f"\nSaved test GIF: {gif_path}")

    ray.shutdown()
