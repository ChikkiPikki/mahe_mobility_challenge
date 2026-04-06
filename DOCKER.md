# Docker Setup — Mini R1 v1 (ROS 2 Jazzy + Gazebo Harmonic)

## Prerequisites

- Docker Engine installed ([docs.docker.com/engine/install](https://docs.docker.com/engine/install/ubuntu/))
- NVIDIA GPU with drivers installed (verify with `nvidia-smi`)

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

Verify: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`

## 2. Build the Image

```bash
docker build -t mini_r1_jazzy .
```

## 3. Run the Container

```bash
xhost +local:docker

docker run -it --rm \
  --hostname openbot \
  --name openbot \
  --runtime=nvidia \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) \
  mini_r1_jazzy
```

| Flag | Purpose |
|------|---------|
| `--runtime=nvidia` | Uses NVIDIA container runtime (mounts GPU drivers + EGL/GL libraries) |
| `--device /dev/dri` | Grants access to the GPU's DRI device |
| `--group-add $(...)` | Adds host's `render` group so the container user can access `/dev/dri` |

## 4. Inside the Container

```bash
ros2 launch mini_r1_v1_bringup bringup.launch.py
```

## Troubleshooting

### `failed to discover GPU vendor from CDI`
Install the NVIDIA Container Toolkit (Step 1).

### `cannot open display`
Run `xhost +local:docker` on the host before starting the container.

### `llvmpipe` in glxinfo (software rendering)
The `--runtime=nvidia` and PRIME env vars in the Dockerfile should handle this. Verify with `glxinfo | grep renderer` inside the container.
