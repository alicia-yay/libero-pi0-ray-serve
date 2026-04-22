"""
=============================================================================
Scenario 2 Demo: Pi0 VLA + LIBERO via Ray Serve
=============================================================================

  python run_demo.py --placeholder       # Quick test with random policy
  python run_demo.py                     # Full pi0 demo
  python run_demo.py --num-workers 3     # Scale sim workers
"""
import ray
from ray import serve
import numpy as np
import time
import json
import os
import builtins
import argparse

os.environ["MUJOCO_GL"] = "egl"
os.environ["DISPLAY"] = ""
builtins.input = lambda *args, **kwargs: "n"


def run_demo(
    use_pi0=True,
    num_workers=3,
    episodes_per_worker=2,
    max_steps=200,
    task_suite="libero_object",
    record_video=True,
    output_dir="/mnt/cluster_storage/libero_demo",
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  SCENARIO 2: Pi0 VLA + LIBERO via Ray Serve")
    print("=" * 70)

    # ── Step 1: Initialize Ray ──
    ray.init(ignore_reinit_error=True)
    print(f"\nRay cluster: {ray.cluster_resources()}")

    # ── Step 2: Deploy policy server ──
    from policy_server import Pi0PolicyServer, PlaceholderPolicyServer

    if use_pi0:
        print("\n[1/4] Deploying Pi0 VLA policy server (~3B params)...")
        app = Pi0PolicyServer.bind(model_id="lerobot/pi0_libero_base")
    else:
        print("\n[1/4] Deploying placeholder policy server (for testing)...")
        app = PlaceholderPolicyServer.bind()

    policy_handle = serve.run(app, name="policy_server")
    print("  Policy server ready!")

    stats = policy_handle.get_stats.remote().result()
    print(f"  Model: {stats['model_id']}")
    print(f"  Params: {stats['num_params_B']:.2f}B")
    print(f"  Device: {stats['device']}")

    # ── Step 3: Get task count ──
    print(f"\n[2/4] Loading LIBERO {task_suite} tasks...")

    @ray.remote
    def get_num_tasks(suite_name):
        import builtins
        builtins.input = lambda *args, **kwargs: "n"
        from libero.libero import benchmark
        bench = benchmark.get_benchmark(suite_name)()
        return bench.get_num_tasks()

    num_tasks = ray.get(get_num_tasks.remote(task_suite))
    print(f"  Found {num_tasks} tasks")

    # ── Step 4: Launch sim workers ──
    print(f"\n[3/4] Launching {num_workers} sim workers...")
    from sim_worker import SimWorker

    workers = []
    for i in range(num_workers):
        task_idx = i % num_tasks
        worker = SimWorker.remote(task_suite=task_suite, task_idx=task_idx)
        workers.append(worker)
        print(f"  Worker {i}: task_idx={task_idx}")

    # ── Step 5: Run episodes ──
    print(f"\n[4/4] Running {episodes_per_worker} episodes per worker...")
    print(f"  Total episodes: {num_workers * episodes_per_worker}")
    print(f"  Max steps per episode: {max_steps}")
    print()

    all_results = []
    t_demo_start = time.time()

    for ep in range(episodes_per_worker):
        print(f"  Episode {ep + 1}/{episodes_per_worker}...")

        futures = [
            worker.run_episode.remote(
                policy_handle,
                max_steps=max_steps,
                record_video=record_video and (ep == 0),
            )
            for worker in workers
        ]

        results = ray.get(futures)
        for i, r in enumerate(results):
            r["worker_id"] = i
            r["episode"] = ep

            if "frames" in r and r["frames"]:
                import imageio
                safe_name = r["task"].replace(" ", "_")[:40]
                gif_path = os.path.join(output_dir, f"worker{i}_{safe_name}.gif")
                imageio.mimsave(gif_path, r["frames"], fps=15, loop=0)
                print(f"    Saved: {gif_path} ({len(r['frames'])} frames)")
                del r["frames"]

            all_results.append(r)
            status = "✓" if r.get("success") else "○"
            print(f"    {status} Worker {i}: {r['task'][:50]}... "
                  f"reward={r['total_reward']:.2f}, "
                  f"steps={r['steps']}, "
                  f"fps={r['fps']:.1f}, "
                  f"policy_lat={r['avg_policy_latency_ms']:.0f}ms")

    demo_elapsed = time.time() - t_demo_start

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  DEMO RESULTS")
    print("=" * 70)

    final_stats = policy_handle.get_stats.remote().result()

    avg_reward = np.mean([r["total_reward"] for r in all_results])
    avg_fps = np.mean([r["fps"] for r in all_results])
    avg_latency = np.mean([r["avg_policy_latency_ms"] for r in all_results])
    success_rate = np.mean([r.get("success", False) for r in all_results]) * 100

    summary = {
        "scenario": "Pi0 VLA + LIBERO via Ray Serve",
        "policy_model": final_stats["model_id"],
        "policy_params_B": final_stats["num_params_B"],
        "num_workers": num_workers,
        "total_episodes": len(all_results),
        "avg_reward": float(avg_reward),
        "success_rate_pct": float(success_rate),
        "avg_sim_fps": float(avg_fps),
        "avg_policy_latency_ms": float(avg_latency),
        "total_policy_calls": final_stats["total_calls"],
        "demo_elapsed_s": float(demo_elapsed),
    }

    print(f"\n  Policy:          {summary['policy_model']}")
    print(f"  Parameters:      {summary['policy_params_B']:.2f}B")
    print(f"  Sim workers:     {summary['num_workers']}")
    print(f"  Total episodes:  {summary['total_episodes']}")
    print(f"  Avg reward:      {summary['avg_reward']:.4f}")
    print(f"  Success rate:    {summary['success_rate_pct']:.0f}%")
    print(f"  Avg sim FPS:     {summary['avg_sim_fps']:.1f}")
    print(f"  Avg policy lat:  {summary['avg_policy_latency_ms']:.0f}ms")
    print(f"  Total calls:     {summary['total_policy_calls']}")
    print(f"  Demo time:       {summary['demo_elapsed_s']:.1f}s")

    results_path = os.path.join(output_dir, "demo_results.json")
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "episodes": all_results}, f, indent=2, default=str)
    print(f"\n  Results: {results_path}")
    print(f"  GIFs:    {output_dir}/")
    print("\n" + "=" * 70)

    for w in workers:
        ray.get(w.close.remote())
    serve.shutdown()
    ray.shutdown()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scenario 2: Pi0 VLA + LIBERO via Ray Serve")
    parser.add_argument("--placeholder", action="store_true", help="Use random policy instead of pi0")
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    run_demo(
        use_pi0=not args.placeholder,
        num_workers=args.num_workers,
        episodes_per_worker=args.episodes,
        max_steps=args.max_steps,
        task_suite=args.suite,
        record_video=not args.no_video,
    )
