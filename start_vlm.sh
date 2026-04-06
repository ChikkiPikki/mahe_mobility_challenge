#!/bin/bash
###############################################################################
# start_vlm.sh — Launch simulation in Docker + VLM brain with NVIDIA API
#
# This runs:
#   1. Docker container with Gazebo + ROS2 simulation
#   2. VLM brain + dashboard on the HOST using NVIDIA NIM cloud API
#
# Prerequisites:
#   - Docker image: docker build -t mini_r1_jazzy .
#   - pip install openai fastapi "uvicorn[standard]"
#   - NVIDIA_API_KEY set in .env or environment
#
# Usage:
#   export NVIDIA_API_KEY=nvapi-xxxxx
#   ./start_vlm.sh
###############################################################################

set -e

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_IMAGE="mini_r1_jazzy"
CONTAINER_NAME="openbot"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

# Load .env if present
if [ -f "$SRC_DIR/.env" ]; then
    set -a; source "$SRC_DIR/.env"; set +a
fi

if [ -z "$NVIDIA_API_KEY" ]; then
    echo -e "${RED}ERROR: NVIDIA_API_KEY not set.${NC}"
    echo "  export NVIDIA_API_KEY=nvapi-xxxxx"
    echo "  Or: echo 'NVIDIA_API_KEY=nvapi-xxxxx' > $SRC_DIR/.env"
    exit 1
fi

echo -e "${CYAN}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Mini R1 — Simulation + VLM Brain (NVIDIA)    ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════╝${NC}"

cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT

# ── Start Docker ────────────────────────────────────────────────────────
echo -e "\n${CYAN}━━━ Starting Docker Container ━━━${NC}"
xhost +local:docker 2>/dev/null || true
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
    --hostname openbot --name "$CONTAINER_NAME" \
    --network=host --runtime=nvidia \
    --env DISPLAY="$DISPLAY" --env QT_X11_NO_MITSHM=1 \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume "$SRC_DIR":/home/dev/ros2_ws/src \
    --device /dev/dri \
    "$DOCKER_IMAGE" \
    bash -c "sudo chown -R dev:dev /home/dev/ros2_ws/src 2>/dev/null; sleep infinity"

sleep 2
echo -e "${GREEN}Container running.${NC}"

# ── Build inside Docker ─────────────────────────────────────────────────
echo -e "\n${CYAN}━━━ Building workspace ━━━${NC}"
docker exec "$CONTAINER_NAME" bash -c "
    source /opt/ros/jazzy/setup.bash
    cd /home/dev/ros2_ws
    colcon build --symlink-install 2>&1 | tail -5
"
echo -e "${GREEN}Build done.${NC}"

# ── Launch simulation in Docker ─────────────────────────────────────────
echo -e "\n${CYAN}━━━ Launching Gazebo + ROS2 ━━━${NC}"
docker exec -d "$CONTAINER_NAME" bash -c "
    source /opt/ros/jazzy/setup.bash
    source /home/dev/ros2_ws/install/setup.bash
    ros2 launch mini_r1_v1_bringup bringup.launch.py
"

echo -e "${YELLOW}Waiting for simulation (15s)...${NC}"
sleep 15

# ── Run VLM brain on HOST ───────────────────────────────────────────────
echo -e "\n${CYAN}━━━ Starting VLM Brain (NVIDIA Cloud) ━━━${NC}"
echo -e "  Provider:  ${GREEN}NVIDIA NIM (qwen/qwen3.5-397b-a17b)${NC}"
echo -e "  Dashboard: ${GREEN}http://localhost:8765${NC}"
echo ""

source /opt/ros/jazzy/setup.bash 2>/dev/null || true
source "$(dirname "$SRC_DIR")/install/setup.bash" 2>/dev/null || true

export LOCAL_VLM_ENABLED=false
export NVIDIA_API_KEY

cd "$SRC_DIR/mini_r1_v1_application/scripts"
python3 vlm_dashboard.py
