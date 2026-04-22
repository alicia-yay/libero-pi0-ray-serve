# Scenario 2: Pi0 VLA + LIBERO via Ray Serve

A **3.5-billion-parameter** Vision-Language-Action model ([pi0](https://huggingface.co/lerobot/pi0_libero_base)) served through Ray Serve, controlling a Franka robot arm in [LIBERO](https://libero-project.github.io/) tabletop manipulation tasks on an Anyscale cluster.

## What This Demo Shows

Modern robot policies like pi0 are **too large to co-locate with simulation** on the same GPU. This demo decouples them using Ray Serve:

```
┌──────────────────────────────────────────────────────────────┐
│                  Ray Cluster · 4× A10G · Anyscale            │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │ SimWorker 1│  │ SimWorker 2│  │ SimWorker 3│  ← LIBERO   │
│  │   A10G     │  │   A10G     │  │   A10G     │    envs     │
│  │  256×256   │  │  256×256   │  │  256×256   │    render    │
│  │  cameras   │  │  cameras   │  │  cameras   │    images    │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘             │
│        │               │               │                     │
│        └───────────────┼───────────────┘                     │
│                        ▼                                     │
│             ┌─────────────────────┐                          │
│             │    Ray Serve RPC    │  ← routing, batching     │
│             └──────────┬──────────┘                          │
│                        ▼                                     │
│             ┌─────────────────────┐                          │
│             │  Pi0 Policy Server  │  ← 3.5B params, fp16    │
│             │       A10G          │     images → 7d actions  │
│             └─────────────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

The code change on the sim side is one line:
```python
# Before (co-located):
actions = policy(obs)

# After (decoupled via Ray Serve):
actions = policy_handle.predict.remote(obs).result()
```

Swap the policy anytime — pi0 today, pi0-FAST tomorrow, your fine-tuned model next week. Same pattern works with any sim (Isaac Lab, MuJoCo, LIBERO) and any policy (pi0, Octo, RT-2).

## Prerequisites

1. **Anyscale account** with access to GPU instances (g5.2xlarge)
2. **HuggingFace account** with an access token → [create one here](https://huggingface.co/settings/tokens)
3. **Accept the PaliGemma license** → [click "Agree" here](https://huggingface.co/google/paligemma-3b-pt-224) (pi0's vision backbone is gated)

## Quick Start

```bash
# 0. Clone into your Anyscale workspace
git clone https://github.com/alicia-yay/libero-pi0-ray-serve.git
cd libero-pi0-ray-serve

# 1. Set your HuggingFace token
export HF_TOKEN=hf_your_token_here

# 2. Patch all worker nodes (groot fix, transformers bypass, HF login)
python patch_workers.py
python set_token.py

# 3. Test LIBERO sim (random actions, saves GIF)
python test_libero.py

# 4. Test full pipeline with placeholder policy (no model download)
python run_demo.py --placeholder --num-workers 3

# 5. Full demo with Pi0 VLA (~6 GB download, 2-3 min to load)
python run_demo.py --num-workers 3 --episodes 1 --max-steps 50
```

## Anyscale Workspace Setup

### Container Image

Build from the included `Containerfile`, or use the pre-built image:
```
anyscale/image/libero-pi0-ray-serve:3
```

Base: `anyscale/ray:2.53.0-slim-py311-cu128` + robosuite + libero + lerobot (--no-deps) + transformers + torch 2.7 CUDA 12.8.

### Cluster Configuration

| Role | Instance | RAM | GPU | Count |
|------|----------|-----|-----|-------|
| Head node | m5.2xlarge | 32 GB | — | 1 |
| GPU workers | g5.2xlarge | 32 GB | A10G 24GB | 4 |

> **Why g5.2xlarge?** Pi0 in float16 is ~6 GB of weights, but `from_pretrained` briefly peaks at ~2× in system RAM. g5.xlarge (16 GB RAM) gets OOM-killed; g5.2xlarge (32 GB) loads cleanly.

## Files

| File | Purpose |
|------|---------|
| `Containerfile` | Docker image definition |
| `patch_workers.py` | **Run first.** Fixes groot, factory.py, transformers check on all workers |
| `set_token.py` | Writes `HF_TOKEN` to all worker nodes |
| `setup.sh` | Alternative setup: patches + dep verification |
| `libero_env.py` | LIBERO env wrapper (headless, monkey-patched, correct BDDL paths) |
| `policy_server.py` | Pi0 via Ray Serve + `PlaceholderPolicyServer` for testing |
| `sim_worker.py` | Ray remote actor: LIBERO sim → camera images → policy → actions → GIFs |
| `run_demo.py` | Orchestrator: deploy policy → launch workers → run episodes → save results |
| `test_libero.py` | Smoke test: one worker, random actions, 50 steps, saves GIF |

## Demo Output

```
Policy:        lerobot/pi0_libero_base  (~3.5B params, float16)
Sim workers:   3 × LIBERO on A10G
Tasks:         libero_object (10 tasks, e.g. "pick up alphabet soup and place in basket")
Observations:  agentview 256×256 RGB + wrist 256×256 RGB + proprio
Actions:       7-dim (delta EE xyz + rotation + gripper)
Output:        GIFs per worker + demo_results.json
```

Run `--placeholder` first to validate the pipeline without downloading the 6 GB model.

## Known Issues & Fixes

All handled by `patch_workers.py`:

| Issue | Fix |
|-------|-----|
| `groot` dataclass crash (Python 3.11) | Delete `groot/` dir, remove imports |
| `factory.py` ImportError on groot | Stub `GrootConfig = type(...)` |
| "incorrect transformer version" | Replace `raise ValueError` with `pass` |
| LIBERO `input()` hangs in Ray workers | `builtins.input = lambda *a, **k: 'n'` |
| Robosuite 1.4 no `action_spec` | Hardcoded `action_dim = 7` |
| Ray Serve `DeploymentResponse` | `.result()` instead of `ray.get()` |
| OOM on g5.xlarge | g5.2xlarge + `dtype=torch.float16` |
| HuggingFace 403 on PaliGemma | Accept license at huggingface.co/google/paligemma-3b-pt-224 |
| Pi0 image key mismatch | `agentview` → `observation.images.image` |

### Open: PaliGemma API Mismatch

`modeling_pi0.py` line 432 calls `self.paligemma.model.get_image_features()`, but no standard `transformers` release (tested 4.51–5.6) has `.model` on `PaliGemmaForConditionalGeneration`. lerobot was built against a custom patched transformers. This is the last issue before end-to-end pi0 inference works. The placeholder demo validates the full architecture without this dependency.

## Comparison with Scenario 1

| | Scenario 1 | Scenario 2 |
|---|---|---|
| Sim | Isaac Lab (GPU-accelerated) | LIBERO / robosuite (MuJoCo) |
| Policy | MLP (~200K params) | Pi0 VLA (~3.5B params) |
| Co-located? | Yes, same GPU | No, separate GPUs |
| Serving | Direct function call | Ray Serve deployment |
| Use case | Robustness sweep | VLA deployment pattern |

## License

Apache 2.0
