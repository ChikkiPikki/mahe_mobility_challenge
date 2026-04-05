# Mini R1 Simulation & Navigation Documentation
*Notes from system bringup and navigation integration.*

## 1. Simulation & World Generation
- **World Generation (CMake/Sed):** The Gazebo simulation relies on procedurally generated `.world` files from YAML via `rmf_building_map_tools`.
- **System Plugins:** To ensure sensors like the IMU and Lidar are active, we inject the required Gazebo system plugins directly into the generated `output.world` using `sed` commands in `mini_r1_v1_description/CMakeLists.txt`. We use an unconditional `add_custom_target(ALL)` to bypass Colcon caching issues.

## 2. Teleop & Physics Stability
- **RViz Teleop Plugin:** The custom WASD teleop panel requires proper registration via `pluginlib_export_plugin_description_file` in CMake and a matching `plugin_description.xml`.
- **Physics Tuning:** The Teleop node max linear and angular velocities were capped at 0.5 m/s and 0.5 rad/s. Setting these too high (e.g., 3.0 m/s) caused physics instability ("flying bot") in Gazebo because of impulsive wheel torques.

## 3. Nav2 Local Costmap Integration
- **Standalone Costmap Node:** A local costmap is run independently via the `nav2_costmap_2d` executable and managed by `nav2_lifecycle_manager`.
- **Parameters (`config/costmap.yaml`):**
  - Configured with `obstacle_layer` and `inflation_layer`.
  - **Raytrace Clearing:** Set `clearing: false`. In a purely local, standalone costmap, setting clearing to true causes obstacles to erase as soon as the robot looks away, creating a visual effect where the entire map appears to "rotate" with the robot. Disabling it leaves a stable trail of obstacles.
  - **Lifecycle Heartbeat:** We set `bond_timeout: 0.0` in the lifecycle manager. The default heartbeat checking conflicts with the initial accumulation of Gazebo Simulation Time during startup, causing a repeating cycle of bond failures.
  - **Topic Remapping:** By default, the node publishes to the root `/costmap`. It was explicitly remapped to `/local_costmap/costmap` in `bringup.launch.py` to match conventional Nav2 namespacing and the RViz Map display.

## 4. Sensor Fusion & EKF
**Problem:** The Gazebo `diff_drive` plugin calculates odometry strictly from wheel joints. If the robot gets stuck but the wheels keep spinning, the odometry falsely reports movement, corrupting the map and position tracking.

**Solution:** Integrate `robot_localization` (EKF Node).
1. **Disabled Gazebo's Direct TF:**
   - Gazebo's `diff_drive` plugin automatically publishes the `odom` → `base_link` transform. Since we cannot modify the locked URDF, we severed the bridge in `mini_r1_v1_gz/config/ros_gz_bridge.yaml` by commenting out the `/r1_mini/odom_tf` → `/tf` route.
   - Non-base joints (like wheels) are completely unaffected because `robot_state_publisher` generates their TF independently via `/joint_states`.
2. **EKF Configuration (`config/ekf.yaml`):**
   - **Odometry (`/r1_mini/odom`):** Trusted purely for linear planar velocities ($v_X, v_Y$).
   - **IMU (`/r1_mini/imu`):** Heavily trusted for absolute Yaw orientation and angular velocity ($\omega_Z$).
   - *Result*: The EKF publishes the true `odom` → `base_link` transform to `/tf`. If the wheels slip while the IMU detects zero physical motion, the fused pose remains stable.
