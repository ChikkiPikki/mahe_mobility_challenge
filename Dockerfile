###############################################################################
# Dockerfile — Mini R1 v1  (ROS 2 Jazzy · Ubuntu 24.04 · Gazebo Harmonic)
#
# Build (from the repo root):
#   docker build -t mini_r1_jazzy .
#
# Run (GPU + display + host network for VLM access):
#   xhost +local:docker
#   docker run -it --rm \
#     --hostname openbot \
#     --name openbot \
#     --network=host \
#     --runtime=nvidia \
#     --env DISPLAY=$DISPLAY \
#     --env QT_X11_NO_MITSHM=1 \
#     --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
#     --volume /path/to/src:/home/dev/ros2_ws/src \
#     --device /dev/dri \
#     mini_r1_jazzy
#
# Or use ./start.sh for one-command launch (starts vLLM + Docker + ROS2)
###############################################################################

FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive

# ── NVIDIA Container Runtime ────────────────────────────────────────────
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV __NV_PRIME_RENDER_OFFLOAD=1
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia

LABEL hostname="openbot"

# ── Add OSRF Gazebo repo (for gz Python bindings not in ROS packages) ──
RUN apt-get update && apt-get install -y --no-install-recommends wget lsb-release gnupg \
    && wget -q https://packages.osrfoundation.org/gazebo.gpg \
       -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/gazebo-stable.list \
    && rm -rf /var/lib/apt/lists/*

# ── System + ROS 2 Jazzy + Gazebo Harmonic packages ────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ── Build tools ──────────────────────────────────────────────────────
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    git \
    wget \
    # ── X11 / rendering support ──────────────────────────────────────────
    libxext6 \
    libxrender1 \
    libxtst6 \
    mesa-utils \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    # ── Gazebo Harmonic (native pairing with Jazzy) ──────────────────────
    ros-jazzy-ros-gz \
    # ── ROS 2 packages ───────────────────────────────────────────────────
    ros-jazzy-twist-stamper \
    ros-jazzy-nav2-costmap-2d \
    ros-jazzy-nav2-lifecycle-manager \
    ros-jazzy-robot-localization \
    ros-jazzy-xacro \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-cv-bridge \
    ros-jazzy-message-filters \
    ros-jazzy-rviz2 \
    ros-jazzy-rviz-common \
    ros-jazzy-rviz-default-plugins \
    ros-jazzy-rmf-building-map-msgs \
    ros-jazzy-rmf-site-map-msgs \
    ros-jazzy-rmf-utils \
    # ── C++ / system libraries ───────────────────────────────────────────
    qtbase5-dev \
    libqt5concurrent5 \
    libyaml-cpp-dev \
    libeigen3-dev \
    libceres-dev \
    libgoogle-glog-dev \
    libproj-dev \
    libsqlite3-dev \
    # ── Python system packages ───────────────────────────────────────────
    python3-opencv \
    python3-numpy \
    python3-yaml \
    python3-shapely \
    python3-pyproj \
    python3-requests \
    python3-rtree \
    python3-fiona \
    # ── Gazebo Harmonic Python bindings (for dynamic_obstacle_node) ──────
    python3-gz-transport13 \
    python3-gz-msgs10 \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages not available via apt ───────────────────────────────
RUN pip3 install --no-cache-dir --break-system-packages \
    scipy \
    Pillow \
    setuptools \
    "numpy<2" \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir --break-system-packages \
    ultralytics \
    opencv-python-headless>=4.10 \
    openai \
    python-dotenv \
    fastapi \
    "uvicorn[standard]"

# ── Initialise rosdep ──────────────────────────────────────────────────
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
        rosdep init; \
    fi && \
    rosdep update --rosdistro=jazzy

# ── Create non-root user ───────────────────────────────────────────────
RUN groupadd -g 109 render 2>/dev/null || true && \
    useradd -m -s /bin/bash -G video,render,dialout dev && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ── Workspace setup ────────────────────────────────────────────────────
USER dev
WORKDIR /home/dev/ros2_ws

COPY --chown=dev:dev . src/

# ── rosdep install ─────────────────────────────────────────────────────
RUN . /opt/ros/jazzy/setup.sh && \
    rosdep install --from-paths src --ignore-src -r -y 2>/dev/null || true

# ── Build (two passes: rmf tools first, then everything) ───────────────
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --symlink-install \
      --packages-up-to rmf_building_map_tools 2>&1 || true

RUN . /opt/ros/jazzy/setup.sh && \
    . install/setup.sh 2>/dev/null || true && \
    colcon build --symlink-install 2>&1 || true

# ── Shell setup ────────────────────────────────────────────────────────
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /home/dev/.bashrc && \
    echo '[ -f /home/dev/ros2_ws/install/setup.bash ] && source /home/dev/ros2_ws/install/setup.bash' >> /home/dev/.bashrc && \
    echo 'export ROS_DOMAIN_ID=1' >> /home/dev/.bashrc && \
    echo 'export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST' >> /home/dev/.bashrc && \
    echo 'export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$HOME/.gz/fuel/fuel.gazebosim.org/openrobotics/models' >> /home/dev/.bashrc && \
    echo 'export GZ_SYSTEM_PLUGIN_PATH=$GZ_SYSTEM_PLUGIN_PATH:/opt/ros/jazzy/lib' >> /home/dev/.bashrc

CMD ["/bin/bash"]
