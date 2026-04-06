#!/bin/bash
###############################################################################
# start_vlm.sh — Launch simulation + VLM brain with NVIDIA API
#
# This runs the simulation in Docker and the VLM brain connects to
# NVIDIA NIM cloud API (no local GPU model needed).
#
# Prerequisites:
#   - NVIDIA_API_KEY set in .env file or environment
#   - Docker image built: docker build -t mini_r1_jazzy .
#
# Usage:
#   ./start_vlm.sh                    # Docker + VLM brain
#   ./start_vlm.sh --native           # No Docker, run directly
#   NVIDIA_API_KEY=nvapi-xxx ./start_vlm.sh  # Pass key via env
###############################################################################

set -e

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
WS_DIR="$(dirname "$SRC_DIR")"
DOCKER_IMAGE="mini_r1_jazzy"
CONTAINER_NAME="openbot"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

NATIVE=false
for arg in "$@"; do
    case $arg in
        --native) NATIVE=true ;;
        --help|-h)
            echo "Usage: ./start_vlm.sh [--native]"
            echo "  --native    Run directly on host (no Docker)"
            echo ""
            echo "Set NVIDIA_API_KEY in .env or environment:"
            echo "  echo 'NVIDIA_API_KEY=nvapi-xxx' > .env"
            exit 0 ;;
    esac
done

# Check for API key
if [ -f "$SRC_DIR/.env" ]; then
    source "$SRC_DIR/.env" 2>/dev/null || true
fi

if [ -z "$NVIDIA_API_KEY" ] && [ -z "$OPENROUTER_API_KEY" ]; then
    echo -e "${RED}ERROR: No VLM API key found.${NC}"
    echo -e "Set one of these:"
    echo -e "  echo 'NVIDIA_API_KEY=nvapi-xxx' > $SRC_DIR/.env"
    echo -e "  echo 'OPENROUTER_API_KEY=sk-or-xxx' > $SRC_DIR/.env"
    echo -e ""
    echo -e "Get a free NVIDIA NIM key at: https://build.nvidia.com/"
    echo -e "Get a free OpenRouter key at: https://openrouter.ai/"
    exit 1
fi

echo -e "${CYAN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Mini R1 — VLM Navigation (Cloud API)    ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════╝${NC}"

if [ -n "$NVIDIA_API_KEY" ]; then
    echo -e "  Provider: ${GREEN}NVIDIA NIM (qwen3.5-397b)${NC}"
fi
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo -e "  Fallback: ${GREEN}OpenRouter (qwen-2.5-vl-72b)${NC}"
fi
echo ""

cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
}
trap cleanup EXIT

if [ "$NATIVE" = true ]; then
    # ── Native mode: run directly on host ──
    echo -e "${CYAN}━━━ Native Mode ━━━${NC}"

    if [ -f /opt/ros/jazzy/setup.bash ]; then
        source /opt/ros/jazzy/setup.bash
    elif [ -f /opt/ros/humble/setup.bash ]; then
        source /opt/ros/humble/setup.bash
    fi

    cd "$WS_DIR"
    echo -e "${YELLOW}Building workspace...${NC}"
    colcon build --symlink-install 2>&1 | tail -5
    source install/setup.bash

    # Export API keys for the brain node
    export NVIDIA_API_KEY="${NVIDIA_API_KEY}"
    export OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"
    export LOCAL_VLM_ENABLED=false

    echo -e "${GREEN}Launching...${NC}"
    ros2 launch mini_r1_v1_bringup bringup.launch.py

else
    # ── Docker mode ──
    echo -e "${CYAN}━━━ Docker Mode ━━━${NC}"

    xhost +local:docker 2>/dev/null || true
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    docker run -it --rm \
        --hostname openbot \
        --name "$CONTAINER_NAME" \
        --network=host \
        --runtime=nvidia \
        --env DISPLAY="$DISPLAY" \
        --env QT_X11_NO_MITSHM=1 \
        --env NVIDIA_API_KEY="${NVIDIA_API_KEY}" \
        --env OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
        --env LOCAL_VLM_ENABLED=false \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --volume "$SRC_DIR":/home/dev/ros2_ws/src \
        --device /dev/dri \
        "$DOCKER_IMAGE" \
        bash -c "
            sudo chown -R dev:dev /home/dev/ros2_ws/src 2>/dev/null || true
            source /opt/ros/jazzy/setup.bash
            cd /home/dev/ros2_ws
            echo '=== Building workspace ==='
            colcon build --symlink-install 2>&1 | tail -5
            source install/setup.bash
            echo '=== Launching with VLM brain ==='
            ros2 launch mini_r1_v1_bringup bringup.launch.py
        "
fi
