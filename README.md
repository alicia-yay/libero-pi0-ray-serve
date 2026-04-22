# Scenario 2: Pi0 VLA + LIBERO via Ray Serve

A ~3B parameter Vision-Language-Action model ([pi0](https://huggingface.co/lerobot/pi0_libero_base)) served through Ray Serve, controlling a Franka arm in [LIBERO](https://libero-project.github.io/) tabletop manipulation tasks.

## Architecture

![Architecture diagram](media/architecture.svg)

The key pattern: the VLA model is **too large to co-locate with each sim worker**, so it lives in a dedicated Ray Serve deployment. Sim workers send camera images + proprio over RPC and receive 7-dim actions back. This is the same pattern you'd use in production with any large policy model.

## Quick Start

```bash
bash setup.sh                            # Fix deps + verify
python test_libero.py                    # Test sim (random actions, GIF)
python run_demo.py --placeholder         # Test pipeline (random policy)
python run_demo.py --num-workers 3       # Full demo with Pi0 VLA
```

## Prerequisites

1. **HuggingFace account** with an access token ([create one here](https://huggingface.co/settings/tokens))
2. **Accept the PaliGemma license** at https://huggingface.co/google/paligemma-3b-pt-224 (required by the pi0 tokenizer)
3. Set your token on all workers:
```bash
   export HF_TOKEN=hf_your_token_here
   python set_token.py  # writes token to all worker nodes
```

## Files

| File | Purpose |
|------|---------|
| `setup.sh` | Fix lerobot groot bug, verify all deps |
| `libero_env.py` | LIBERO env wrapper |
| `policy_server.py` | Pi0 VLA via Ray Serve (+ placeholder mode) |
| `sim_worker.py` | Ray remote sim worker |
| `run_demo.py` | Full demo orchestrator |
| `test_libero.py` | Quick sim test with random actions |

## Instance Requirements

| Role | Instance | RAM | VRAM |
|------|----------|-----|------|
| Policy server | g5.2xlarge | 32 GB | 24 GB |
| Sim workers (×3) | g5.2xlarge | 32 GB | 24 GB |
| Head node | m5.2xlarge | 32 GB | — |

> **Why g5.2xlarge?** Pi0 in float16 is ~6 GB of weights, but loading via HuggingFace briefly peaks at ~2× that in system RAM. g5.xlarge (16 GB RAM) OOMs during load; g5.2xlarge (32 GB RAM) is safe.

## Known Issues

- `lerobot` groot module is broken on Python 3.11 → `setup.sh` removes it and patches `factory.py` on all worker nodes.
- LIBERO prompts for a dataset path interactively → monkey-patched with `builtins.input` in every Ray remote function.
- Robosuite 1.4 has no `action_spec` → action dim is hardcoded to `7` (Franka always outputs 7-dim delta EE + gripper).
- Pi0 `torch_dtype` kwarg: pass `dtype=torch.float16` (not the deprecated `torch_dtype=`) to load weights directly in float16 and avoid the float32 intermediate that causes OOM.
- Ray Serve 2.53 returns `DeploymentResponse`, not `ObjectRef` → use `.result()` instead of `ray.get()`.

## What the Demo Shows

```
Policy:        lerobot/pi0_libero_base  (~3B params, float16)
Sim workers:   3 × LIBERO on A10G
Tasks:         libero_object (10 tasks, e.g. "pick up alphabet soup and place in basket")
Observations:  agentview 256×256 RGB + wrist 256×256 RGB + proprio
Actions:       7-dim (Δ end-effector xyz, rotation, gripper)
Output:        GIFs per worker + demo_results.json with timing stats
```

Run the placeholder first (`--placeholder`) to validate the pipeline without downloading the ~6 GB model. Once that passes, drop `--placeholder` for real pi0 inference.
