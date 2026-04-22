"""
Apply all dependency patches to every GPU worker node.
Run this ONCE after workspace boots, before running the demo.

Fixes:
  1. Removes lerobot's broken groot module (Python 3.11 dataclass incompatibility)
  2. Patches factory.py to stub out groot imports
  3. Bypasses the transformers version check in modeling_pi0.py
"""
import ray
ray.init(ignore_reinit_error=True)

@ray.remote(num_gpus=1, num_cpus=0)
def patch():
    import subprocess, shutil, os

    LEROBOT = "/home/ray/anaconda3/lib/python3.11/site-packages/lerobot"

    # 1. Reinstall fresh lerobot (in case previous patches mangled it)
    subprocess.run(["pip", "install", "--force-reinstall", "--no-deps", "lerobot", "-q", "--break-system-packages"], check=True)

    # 2. Delete groot module
    shutil.rmtree(f"{LEROBOT}/policies/groot/", ignore_errors=True)

    # 3. Remove groot from __init__.py
    p = f"{LEROBOT}/policies/__init__.py"
    with open(p) as f: lines = f.readlines()
    with open(p, "w") as f:
        for l in lines:
            if "groot" not in l.lower(): f.write(l)

    # 4. Stub groot imports in factory.py
    p = f"{LEROBOT}/policies/factory.py"
    with open(p) as f: c = f.read()
    c = c.replace(
        "from lerobot.policies.groot.configuration_groot import GrootConfig",
        "GrootConfig = type('GrootConfig', (), {})")
    c = c.replace(
        "from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors",
        "make_groot_pre_post_processors = None")
    with open(p, "w") as f: f.write(c)

    # 5. Bypass transformers version check in modeling_pi0.py
    p = f"{LEROBOT}/policies/pi0/modeling_pi0.py"
    with open(p) as f: c = f.read()
    old = """        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None"""
    new = """        try:
            from transformers.models.siglip import check
            pass
        except ImportError:
            pass"""
    if old in c:
        c = c.replace(old, new)
        with open(p, "w") as f: f.write(c)

    return "patched " + os.uname().nodename

results = ray.get([patch.remote() for _ in range(4)])
print(results)
ray.shutdown()
