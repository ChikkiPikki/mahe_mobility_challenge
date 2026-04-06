#!/bin/bash
###############################################################################
# dev.sh — Development launcher (no Docker, runs directly on host)
#
# Usage:
#   ./dev.sh              # build + launch full stack
#   ./dev.sh --vlm        # also start vLLM server
#   ./dev.sh --build-only # just build, don't launch
#   ./dev.sh --no-build   # skip build, just launch
###############################################################################

set -e

WS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$WS_DIR/src"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

START_VLM=false
BUILD=true
LAUNCH=true

for arg in "$@"; do
    case $arg in
        --vlm)        START_VLM=true ;;
        --build-only) LAUNCH=false ;;
        --no-build)   BUILD=false ;;
        --help|-h)
            echo "Usage: ./dev.sh [--vlm] [--build-only] [--no-build]"
            echo "  --vlm         Also start local vLLM server"
            echo "  --build-only  Build workspace, don't launch"
            echo "  --no-build    Skip build, just launch"
            exit 0 ;;
    esac
done

VLLM_PID=""
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null
        wait "$VLLM_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

# Source ROS2
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
elif [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
else
    echo -e "${RED}No ROS2 installation found${NC}"
    exit 1
fi

cd "$WS_DIR"

# Build
if [ "$BUILD" = true ]; then
    echo -e "${CYAN}━━━ Building workspace ━━━${NC}"
    colcon build --symlink-install 2>&1 | tail -10
    echo -e "${GREEN}Build complete.${NC}"
fi

# Source workspace
if [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
fi

if [ "$LAUNCH" = false ]; then
    echo -e "${GREEN}Build only — done.${NC}"
    exit 0
fi

# Optional: start vLLM
if [ "$START_VLM" = true ]; then
    echo -e "${CYAN}━━━ Starting vLLM Server ━━━${NC}"
    if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
        echo -e "${GREEN}vLLM already running on :8000${NC}"
    else
        VLLM_PYTHON=""
        if command -v vllm &>/dev/null; then
            VLLM_PYTHON="$(command -v python3)"
        elif [ -f "$HOME/anaconda3/bin/python" ]; then
            VLLM_PYTHON="$HOME/anaconda3/bin/python"
        fi
        if [ -n "$VLLM_PYTHON" ]; then
            $VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
                --model Qwen/Qwen2.5-VL-3B-Instruct \
                --max-model-len 1024 \
                --gpu-memory-utilization 0.90 \
                --port 8000 \
                --trust-remote-code \
                --dtype half \
                --enforce-eager \
                2>&1 | sed 's/^/  [vLLM] /' &
            VLLM_PID=$!
            echo -e "${YELLOW}Waiting for vLLM...${NC}"
            for i in $(seq 1 60); do
                curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && break
                sleep 2
            done
            echo -e "${GREEN}vLLM ready.${NC}"
        else
            echo -e "${RED}vLLM not found. Skipping.${NC}"
        fi
    fi
fi

# Launch
echo -e "${CYAN}━━━ Launching ROS2 Stack ━━━${NC}"
ros2 launch mini_r1_v1_bringup bringup.launch.py
