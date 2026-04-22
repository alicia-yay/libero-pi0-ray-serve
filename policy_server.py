"""
Pi0 VLA Policy Server via Ray Serve.

Architecture:
  [Sim Worker 1] ──┐
  [Sim Worker 2] ──┤──> [Ray Serve: Pi0 Policy] ──> actions
  [Sim Worker 3] ──┘

NOTE: lerobot's groot module is broken on Python 3.11. Fix before importing:
  rm -rf /path/to/lerobot/policies/groot/
  sed -i '/groot/d' /path/to/lerobot/policies/__init__.py
"""
import ray
from ray import serve
import torch
import numpy as np
import time
import builtins

builtins.input = lambda *args, **kwargs: "n"


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=16,
)
class Pi0PolicyServer:
    """Serves pi0 VLA model for LIBERO tasks."""

    def __init__(self, model_id="lerobot/pi0_libero_base"):
        import builtins
        builtins.input = lambda *args, **kwargs: "n"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self._load_model()

    def _load_model(self):
        print(f"[Pi0Server] Loading {self.model_id} on {self.device}...")
        t0 = time.time()

        from lerobot.policies.pi0 import PI0Policy
        from lerobot.policies.factory import make_pre_post_processors

        self.policy = PI0Policy.from_pretrained(self.model_id, dtype=torch.float16).to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            self.model_id,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        num_params = sum(p.numel() for p in self.policy.parameters())
        load_time = time.time() - t0
        print(f"[Pi0Server] Loaded in {load_time:.1f}s")
        print(f"[Pi0Server] Parameters: {num_params / 1e9:.2f}B")

        self.num_params = num_params
        self.load_time = load_time
        self._call_count = 0
        self._total_latency = 0.0

    async def predict(self, obs_dict: dict) -> dict:
        t0 = time.time()

        with torch.no_grad():
            agentview = torch.from_numpy(obs_dict["agentview_image"]).permute(2, 0, 1).float() / 255.0
            wrist = torch.from_numpy(obs_dict["wrist_image"]).permute(2, 0, 1).float() / 255.0

            # Build robot state vector from proprio
            import numpy as np
            robot_state = np.concatenate([
                obs_dict.get("robot_state", np.zeros(3)),
                obs_dict.get("gripper_state", np.zeros(2)),
            ]).astype(np.float32)
            state_tensor = torch.from_numpy(robot_state.copy()).unsqueeze(0).to(self.device)

            batch = {
                "observation.images.image": agentview.unsqueeze(0).to(self.device),
                "observation.images.image2": wrist.unsqueeze(0).to(self.device),
                "observation.state": state_tensor,
                "task": [obs_dict.get("language_instruction", "pick up the object")],
            }

            batch = self.preprocess(batch)
            output = self.policy.select_action(batch)
            output = self.postprocess(output)
            action = output["action"].cpu().numpy().flatten()

        latency_ms = (time.time() - t0) * 1000
        self._call_count += 1
        self._total_latency += latency_ms

        return {"action": action, "latency_ms": latency_ms}

    async def get_stats(self) -> dict:
        avg_latency = self._total_latency / max(self._call_count, 1)
        return {
            "model_id": self.model_id,
            "num_params_B": self.num_params / 1e9,
            "device": str(self.device),
            "total_calls": self._call_count,
            "avg_latency_ms": avg_latency,
        }


@serve.deployment(ray_actor_options={"num_gpus": 0})
class PlaceholderPolicyServer:
    """Random actions for testing the architecture without pi0."""

    def __init__(self):
        print("[PlaceholderPolicy] Ready (random actions, no GPU)")
        self._call_count = 0
        self._total_latency = 0.0

    async def predict(self, obs_dict: dict) -> dict:
        t0 = time.time()
        action = np.random.uniform(-0.05, 0.05, size=(7,)).astype(np.float32)
        latency_ms = (time.time() - t0) * 1000
        self._call_count += 1
        self._total_latency += latency_ms
        return {"action": action, "latency_ms": latency_ms}

    async def get_stats(self) -> dict:
        avg_latency = self._total_latency / max(self._call_count, 1)
        return {
            "model_id": "placeholder-random",
            "num_params_B": 0.0,
            "device": "cpu",
            "total_calls": self._call_count,
            "avg_latency_ms": avg_latency,
        }
