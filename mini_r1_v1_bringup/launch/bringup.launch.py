import os
import math
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    # 1. Include the Gazebo Sim Launch
    sim_package_share = get_package_share_directory("mini_r1_v1_gz")
    description_package_share = get_package_share_directory("mini_r1_v1_description")
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sim_package_share, "launch", "sim.launch.py")
        )
    )

    # Compute spawn and goal points in world coordinates from YAML (same logic as sim.launch.py)
    spawn_x = 0.0
    spawn_y = 0.0
    goal_x = 0.0
    goal_y = 0.0
    try:
        yaml_path = os.path.join(description_package_share, 'worlds', 'multi_floor_college.building.yaml')
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        scale = 1.0
        measurements = data['levels']['floor_0']['measurements'][0]
        v1_idx, v2_idx = measurements[0], measurements[1]
        distance_m = measurements[2]['distance'][1]
        v1 = data['levels']['floor_0']['vertices'][v1_idx]
        v2 = data['levels']['floor_0']['vertices'][v2_idx]
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        distance_px = math.hypot(dx, dy)
        scale = distance_m / distance_px
        for v in data['levels']['floor_0']['vertices']:
            if len(v) >= 4 and isinstance(v[3], str):
                label = str(v[3])
                if label.startswith('spawn'):
                    spawn_x = v[0] * scale
                    spawn_y = -v[1] * scale
                elif label == 'goal':
                    goal_x = v[0] * scale
                    goal_y = -v[1] * scale
    except Exception as e:
        print(f'Warning: could not compute spawn/goal offset: {e}')

    # 2. Standard tf2_ros mappings connecting Gazebo's implicitly generated namespaces back to our internal URDF labels
    tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_lidar',
        arguments=['0', '0', '0', '0', '0', '0', 'LIDAR', 'mini_r1/base_link/lidar']
    )
    
    tf_cam = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_cam',
        arguments=['0', '0', '0', '0', '0', '0', 'CAM', 'mini_r1/base_link/camera']
    )
    
    tf_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_imu',
        arguments=['0', '0', '0', '0', '0', '0', 'IMU', 'mini_r1/base_link/imu']
    )

    # 3. Nav2 Local Costmap (standalone)
    costmap_params = os.path.join(sim_package_share, 'config', 'costmap.yaml')

    local_costmap = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='local_costmap',
        output='screen',
        parameters=[costmap_params],
        remappings=[
            ('costmap', 'local_costmap/costmap'),
            ('costmap_updates', 'local_costmap/costmap_updates'),
            ('costmap_raw', 'local_costmap/costmap_raw'),
            ('costmap_raw_updates', 'local_costmap/costmap_raw_updates')
        ]
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_costmap',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['local_costmap'],
            'use_sim_time': True,
            'bond_timeout': 0.0,
        }],
    )

    # 4. EKF Node for Odometry + IMU fusion
    ekf_params = os.path.join(get_package_share_directory('mini_r1_v1_bringup'), 'config', 'ekf.yaml')
    
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_params],
    )

    # 5. Application Node: Marker Detector (with config + spawn offset)
    app_package_share = get_package_share_directory('mini_r1_v1_application')
    marker_detector_config = os.path.join(app_package_share, 'config', 'marker_detector.yaml')

    marker_detector = Node(
        package='mini_r1_v1_application',
        executable='marker_detector_node.py',
        name='marker_detector_node',
        output='screen',
        parameters=[
            marker_detector_config,
            {
                'spawn_x': spawn_x,
                'spawn_y': spawn_y,
            }
        ]
    )

    # 6. Mission Zone Node: Start/Goal cylinders (RViz) + goal proximity check
    mission_zone = Node(
        package='mini_r1_v1_application',
        executable='mission_zone_node.py',
        name='mission_zone_node',
        output='screen',
        parameters=[{
            'spawn_x': spawn_x,
            'spawn_y': spawn_y,
            'goal_x': goal_x,
            'goal_y': goal_y,
        }]
    )

    return LaunchDescription([
        sim_launch,
        tf_lidar,
        tf_cam,
        tf_imu,
        local_costmap,
        lifecycle_manager,
        ekf_node,
        marker_detector,
        mission_zone,
    ])

