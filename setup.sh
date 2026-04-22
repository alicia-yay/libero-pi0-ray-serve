#!/bin/bash
# Setup for LIBERO + Pi0 + Ray Serve workspace
set -e

echo "=== LIBERO + Pi0 + Ray Serve Setup ==="

# 1. Fix lerobot groot module (broken on Python 3.11)
echo ""
echo "Patching lerobot (removing broken groot module)..."
LEROBOT_DIR=$(python -c "import lerobot; import os; print(os.path.dirname(lerobot.__file__))")
rm -rf "${LEROBOT_DIR}/policies/groot/" 2>/dev/null || true
sed -i '/groot/d' "${LEROBOT_DIR}/policies/__init__.py" 2>/dev/null || true
echo "  Done"

# 2. Create LIBERO config (skip interactive prompt)
echo ""
echo "Creating LIBERO config..."
python -c "
import os, json
config_dir = os.path.expanduser('~/.libero')
os.makedirs(config_dir, exist_ok=True)
# Just create the marker so LIBERO doesn't prompt
"
echo "  Done"

# 3. Create dirs
mkdir -p /mnt/cluster_storage/libero_demo
mkdir -p /mnt/cluster_storage/libero_data

# 4. Verify installs
echo ""
echo "Verifying installs..."
python -c "import torch; print(f'  PyTorch: OK (CUDA: {torch.cuda.is_available()})')"
python -c "import ray; print(f'  Ray: OK ({ray.__version__})')"
python -c "import robosuite; print(f'  Robosuite: OK ({robosuite.__version__})')" 2>/dev/null
python -c "import builtins; builtins.input = lambda *a,**k: 'n'; import libero; print('  LIBERO: OK')"
python -c "from lerobot.policies.pi0 import PI0Policy; print('  LeRobot Pi0: OK')"
python -c "import transformers; print(f'  Transformers: OK ({transformers.__version__})')"

# 5. Check cluster
echo ""
echo "Cluster status:"
ray status 2>/dev/null | head -10 || echo "  Ray not initialized yet"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Quick start:"
echo "  1. python test_libero.py                     # Test sim (random actions, GIF)"
echo "  2. python run_demo.py --placeholder          # Test pipeline (random policy)"
echo "  3. python run_demo.py                        # Full demo with Pi0 VLA"
