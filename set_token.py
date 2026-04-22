"""Write HF token to all worker nodes via Ray."""
import ray, os

ray.init(ignore_reinit_error=True)
TOKEN = os.environ.get("HF_TOKEN", "")
if not TOKEN:
    print("ERROR: Set HF_TOKEN first: export HF_TOKEN=hf_your_token_here")
    exit(1)

@ray.remote(num_gpus=1, num_cpus=0)
def set_token(token):
    import os
    os.makedirs(os.path.expanduser("~/.cache/huggingface"), exist_ok=True)
    with open(os.path.expanduser("~/.cache/huggingface/token"), "w") as f:
        f.write(token)
    return "done on " + os.uname().nodename

results = ray.get([set_token.remote(TOKEN) for _ in range(4)])
print("Token set on:", results)
ray.shutdown()
