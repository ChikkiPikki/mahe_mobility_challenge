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
OLLAMA_MODEL="qwen2.5vl:3b"
OLLAMA_PORT=11434
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
start_vlm() {
    echo -e "${CYAN}━━━ Starting Ollama VLM Server ━━━${NC}"
    echo -e "  Model: ${GREEN}$OLLAMA_MODEL${NC}"
    echo -e "  Port:  ${GREEN}$OLLAMA_PORT${NC}"

    # Check if Ollama is installed
    if ! command -v ollama &>/dev/null; then
        echo -e "${RED}Ollama not found. Installing...${NC}"
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    # Check if already serving
    if curl -s "http://localhost:$OLLAMA_PORT/v1/models" >/dev/null 2>&1; then
        echo -e "${GREEN}Ollama already running on port $OLLAMA_PORT${NC}"
    else
        echo -e "${YELLOW}Starting Ollama serve...${NC}"
        ollama serve 2>&1 | sed 's/^/  [Ollama] /' &
        VLLM_PID=$!
        sleep 3
    fi

    # Pull model if needed
    echo -e "${YELLOW}Ensuring model $OLLAMA_MODEL is available...${NC}"
    ollama pull "$OLLAMA_MODEL" 2>&1 | sed 's/^/  [Ollama] /'

    # Verify
    if curl -s "http://localhost:$OLLAMA_PORT/v1/models" >/dev/null 2>&1; then
        echo -e "${GREEN}Ollama ready on http://localhost:$OLLAMA_PORT/v1${NC}"
    else
        echo -e "${RED}ERROR: Ollama failed to start${NC}"
        exit 1
    fi
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
    start_vlm
    echo -e "\n${GREEN}vLLM server running. Press Ctrl+C to stop.${NC}"
    wait "$VLLM_PID"
    exit 0
fi

if [ "$NO_VLM" = false ]; then
    start_vlm
fi

start_docker
build_and_launch
