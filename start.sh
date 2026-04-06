#!/bin/bash
###############################################################################
# start.sh — Launch the full Mini R1 stack (vLLM + Docker + ROS2)
#
# This script:
#   1. Starts the vLLM server on the host (Qwen2.5-VL-3B vision model)
#   2. Waits for the model to be ready
#   3. Starts the Docker container with Gazebo + ROS2 + navigation
#   4. Builds the workspace and launches the full bringup
#
# Prerequisites:
#   - vLLM installed: pip install vllm qwen-vl-utils
#   - Model downloaded: huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
#   - Docker image built: docker build -t mini_r1_jazzy .
#   - NVIDIA Container Toolkit installed
#
# Usage:
#   ./start.sh              # full stack (vLLM + Docker + ROS2)
#   ./start.sh --no-vlm     # Docker + ROS2 only (no local model)
#   ./start.sh --vlm-only   # start vLLM server only
###############################################################################

set -e

# ── Configuration ───────────────────────────────────────────────────────
VLLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
VLLM_PORT=8000
VLLM_MAX_LEN=1024
VLLM_GPU_UTIL=0.90
DOCKER_IMAGE="mini_r1_jazzy"
CONTAINER_NAME="openbot"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
WS_DIR="$(dirname "$SRC_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Parse arguments ─────────────────────────────────────────────────────
NO_VLM=false
VLM_ONLY=false
for arg in "$@"; do
    case $arg in
        --no-vlm)   NO_VLM=true ;;
        --vlm-only) VLM_ONLY=true ;;
        --help|-h)
            echo "Usage: ./start.sh [--no-vlm | --vlm-only]"
            echo "  --no-vlm    Skip vLLM server, run Docker + ROS2 only"
            echo "  --vlm-only  Start vLLM server only, no Docker"
            exit 0 ;;
    esac
done

# ── Cleanup on exit ─────────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo -e "${CYAN}Stopping vLLM server (PID $VLLM_PID)...${NC}"
        kill "$VLLM_PID" 2>/dev/null
        wait "$VLLM_PID" 2>/dev/null
    fi
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT

# ── Step 1: Start vLLM Server ──────────────────────────────────────────
start_vllm() {
    echo -e "${CYAN}━━━ Starting vLLM Server ━━━${NC}"
    echo -e "  Model:    ${GREEN}$VLLM_MODEL${NC}"
    echo -e "  Port:     ${GREEN}$VLLM_PORT${NC}"
    echo -e "  GPU util: ${GREEN}$VLLM_GPU_UTIL${NC}"

    # Check if already running
    if curl -s "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
        echo -e "${GREEN}vLLM already running on port $VLLM_PORT${NC}"
        return 0
    fi

    # Find vllm executable
    VLLM_BIN=""
    if command -v vllm &>/dev/null; then
        VLLM_PYTHON="$(command -v python3)"
    elif [ -f "$HOME/anaconda3/bin/python" ]; then
        VLLM_PYTHON="$HOME/anaconda3/bin/python"
    elif [ -f "$HOME/miniconda3/bin/python" ]; then
        VLLM_PYTHON="$HOME/miniconda3/bin/python"
    else
        echo -e "${RED}ERROR: Cannot find vllm. Install with: pip install vllm qwen-vl-utils${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Starting vLLM in background...${NC}"
    $VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --max-model-len "$VLLM_MAX_LEN" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" \
        --port "$VLLM_PORT" \
        --trust-remote-code \
        --dtype half \
        --enforce-eager \
        2>&1 | sed 's/^/  [vLLM] /' &
    VLLM_PID=$!

    # Wait for server to be ready
    echo -e "${YELLOW}Waiting for vLLM to load model (this may take 30-60s)...${NC}"
    for i in $(seq 1 120); do
        if curl -s "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
            echo -e "${GREEN}vLLM ready on http://localhost:$VLLM_PORT/v1${NC}"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo -e "${RED}ERROR: vLLM process died. Check logs above.${NC}"
            exit 1
        fi
        sleep 2
    done
    echo -e "${RED}ERROR: vLLM failed to start within 240s${NC}"
    exit 1
}

# ── Step 2: Start Docker Container ─────────────────────────────────────
start_docker() {
    echo -e "\n${CYAN}━━━ Starting Docker Container ━━━${NC}"

    # Allow X11 access
    xhost +local:docker 2>/dev/null || true

    # Remove existing container if any
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    echo -e "  Image:     ${GREEN}$DOCKER_IMAGE${NC}"
    echo -e "  Container: ${GREEN}$CONTAINER_NAME${NC}"
    echo -e "  Network:   ${GREEN}host (shares host network for vLLM access)${NC}"
    echo -e "  Source:    ${GREEN}$SRC_DIR → /home/dev/ros2_ws/src${NC}"

    docker run -d \
        --hostname openbot \
        --name "$CONTAINER_NAME" \
        --network=host \
        --runtime=nvidia \
        --env DISPLAY="$DISPLAY" \
        --env QT_X11_NO_MITSHM=1 \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --volume "$SRC_DIR":/home/dev/ros2_ws/src \
        --device /dev/dri \
        "$DOCKER_IMAGE" \
        bash -c "
            sudo chown -R dev:dev /home/dev/ros2_ws/src 2>/dev/null || true
            sleep infinity
        "

    # Wait for container to be ready
    sleep 2
    if ! docker ps --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
        echo -e "${RED}ERROR: Container failed to start${NC}"
        docker logs "$CONTAINER_NAME" 2>&1 | tail -10
        exit 1
    fi
    echo -e "${GREEN}Container running.${NC}"
}

# ── Step 3: Build and Launch inside Docker ──────────────────────────────
build_and_launch() {
    echo -e "\n${CYAN}━━━ Building & Launching ROS2 Stack ━━━${NC}"

    docker exec -it "$CONTAINER_NAME" bash -c "
        source /opt/ros/jazzy/setup.bash
        cd /home/dev/ros2_ws

        echo '=== Building workspace ==='
        colcon build --symlink-install 2>&1 | tail -5
        source install/setup.bash

        echo ''
        echo '=== Launching bringup ==='
        ros2 launch mini_r1_v1_bringup bringup.launch.py
    "
}

# ── Main ────────────────────────────────────────────────────────────────
echo -e "${CYAN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     Mini R1 v1 — Full Stack Launcher      ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════╝${NC}"
echo ""

if [ "$VLM_ONLY" = true ]; then
    start_vllm
    echo -e "\n${GREEN}vLLM server running. Press Ctrl+C to stop.${NC}"
    wait "$VLLM_PID"
    exit 0
fi

if [ "$NO_VLM" = false ]; then
    start_vllm
fi

start_docker
build_and_launch
