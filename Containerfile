FROM anyscale/ray:2.53.0-slim-py311-cu128
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl cmake libgl1 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*
USER ray
RUN python -m pip install --no-cache-dir \
    "torch==2.7.0" "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu128
RUN python -m pip install --no-cache-dir \
    robosuite libero
RUN python -m pip install --no-cache-dir --no-deps lerobot
RUN python -m pip install --no-cache-dir \
    transformers accelerate diffusers einops safetensors \
    huggingface_hub datasets draccus deepdiff jsonlines \
    pyserial av matplotlib plotly imageio
WORKDIR /home/ray/default
