# Docker + VLM Setup — Mini R1 v1

## Quick Start (one command)

```bash
./start.sh
```

This starts the vLLM server on the host, launches the Docker container, builds the workspace, and runs the full stack. See below for manual setup.

---

## Prerequisites

- Docker Engine ([docs.docker.com/engine/install](https://docs.docker.com/engine/install/ubuntu/))
- NVIDIA GPU with drivers (`nvidia-smi`)
- NVIDIA Container Toolkit (see below)
- Python 3.10+ with pip on the host

## 1. Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 2. Set Up Local VLM (on host, not Docker)

The VLM runs on the **host machine** so it doesn't compete with Gazebo for GPU memory. The Docker container connects to it via `--network=host`.

```bash
# Install vLLM (one-time)
pip install vllm qwen-vl-utils

# Download the model (one-time, ~7GB)
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct

# Start the server (every session)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.5 \
  --port 8000 \
  --trust-remote-code

# Verify:
curl http://localhost:8000/v1/models
```

The server provides an OpenAI-compatible API at `http://localhost:8000/v1`.

### Using Cloud APIs Instead (no local GPU needed)

Create a `.env` file in the repo root:
```bash
# NVIDIA NIM (recommended):
NVIDIA_API_KEY=nvapi-xxxxxxxx

# OR OpenRouter (free tier):
OPENROUTER_API_KEY=sk-or-xxxxxxxx

# Disable local VLM:
LOCAL_VLM_ENABLED=false
```

## 3. Build the Docker Image

```bash
docker build -t mini_r1_jazzy .
```

## 4. Run the Container

```bash
xhost +local:docker

docker run -it --rm \
  --hostname openbot \
  --name openbot \
  --network=host \
  --runtime=nvidia \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume $(pwd):/home/dev/ros2_ws/src \
  --device /dev/dri \
  mini_r1_jazzy
```

**`--network=host`** is critical — it lets the container access the host's vLLM server at `localhost:8000`.

## 5. Inside the Container

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch mini_r1_v1_bringup bringup.launch.py
```

## 6. VLM Dashboard

Open `http://localhost:8765` in a browser to see real-time VLM reasoning, tool calls, and robot state.

## Troubleshooting

### `failed to discover GPU vendor from CDI`
Install the NVIDIA Container Toolkit (Step 1).

### `cannot open display`
Run `xhost +local:docker` on the host.

### VLM not connecting
Verify vLLM is running: `curl http://localhost:8000/v1/models`
Container must use `--network=host`.

### `llvmpipe` in glxinfo
The `--runtime=nvidia` and PRIME env vars handle this. Verify: `glxinfo | grep renderer`
