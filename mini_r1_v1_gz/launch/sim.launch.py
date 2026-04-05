import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

from launch.launch_description_sources import PythonLaunchDescriptionSource

def launch_setup(context, *args, **kwargs):
    description_package_share = get_package_share_directory("mini_r1_v1_description")
    yaml_path = os.path.join(description_package_share, 'worlds', 'multi_floor_college.building.yaml')
    
    scale = 0.0096932  # Default fallback scale
    spawn_x = 0.0
    spawn_y = 0.0
    spawn_yaw = 0.0

    try:
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Calculate scale
            try:
                measurements = data['levels']['floor_0']['measurements'][0]
                v1_idx = measurements[0]
                v2_idx = measurements[1]
                distance_m = measurements[2]['distance'][1]
                
                v1 = data['levels']['floor_0']['vertices'][v1_idx]
                v2 = data['levels']['floor_0']['vertices'][v2_idx]
                
                dx = v2[0] - v1[0]
                dy = v2[1] - v1[1]
                distance_px = (dx**2 + dy**2)**0.5
                scale = distance_m / distance_px
            except Exception:
                pass
            
            # Find spawn vertex
            for v in data['levels']['floor_0']['vertices']:
                if len(v) >= 4 and isinstance(v[3], str) and str(v[3]).startswith("spawn"):
                    spawn_x = v[0] * scale
                    spawn_y = -v[1] * scale
                    
                    parts = str(v[3]).split("_")
                    if len(parts) >= 2:
                        try:
                            import math
                            spawn_yaw = float(parts[1]) * math.pi / 180.0
                        except ValueError:
                            pass
                    break
    except Exception as e:
        print(f"Error parsing yaml: {e}")

    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-topic", 'robot_description',
                   '-name', 'mini_r1',
                   '-x', str(spawn_x),
                   '-y', str(spawn_y),
                   '-z', '0.07',
                   '-Y', str(spawn_yaw)],
        output="screen"
    )
    return [spawn_entity]


def generate_launch_description():
    description_package = "mini_r1_v1_description"
    simulation_package = "mini_r1_v1_gz"

    description_package_share = get_package_share_directory(description_package)
    simulation_package_share = get_package_share_directory(simulation_package)

    fuel_cache = os.path.expanduser("~/.gz/fuel/fuel.gazebosim.org/openrobotics/models")
    world_models = os.path.join(description_package_share, "worlds", "models")
    ros_lib = os.path.join('/opt/ros', os.environ.get('ROS_DISTRO', 'jazzy'), 'lib')

    gz_sim_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    os.environ['GZ_SIM_RESOURCE_PATH'] = f"{fuel_cache}:{world_models}:{gz_sim_resource_path}"

    gz_system_plugin_path = os.environ.get("GZ_SYSTEM_PLUGIN_PATH", "")
    os.environ['GZ_SYSTEM_PLUGIN_PATH'] = f"{ros_lib}:{gz_system_plugin_path}"

    default_world = os.path.join(description_package_share, 'worlds', 'output.world')
    world_path = LaunchConfiguration('world')

    rsp = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        os.path.join(description_package_share, "launch", "rsp.launch.py")
                    ]
                ),
                launch_arguments={"use_sim_time": "true", 'use_control': "true"}.items()
            )
    gz = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [
                    os.path.join(get_package_share_directory("ros_gz_sim"),'launch', 'gz_sim.launch.py')
                ]
            ),
            launch_arguments={"gz_args": ['-r ', world_path], 'on_exit_shutdown': 'true'}.items(),
    )
    
    stamper = Node(
                package="twist_stamper",
                executable="twist_stamper",
                remappings=[
                    ('cmd_vel_in', 'cmd_vel'),
                    ('cmd_vel_out', 'cmd_vel_stamped'),
                ],
    )
    bridge_params = os.path.join(get_package_share_directory(simulation_package), 'config', 'ros_gz_bridge.yaml')
    ros_gz_bridge = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                '--ros-args',
                '-p',
                f'config_file:={bridge_params}'
            ]
        )
    launch_args = [
        DeclareLaunchArgument(
            name="world",
            default_value=default_world,
            description="Enter the absolute path to the world in which the robot is to be spawned"
        )
    ]

    rviz_config_file = os.path.join(simulation_package_share, 'rviz', 'sim.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([ 
        *launch_args,
        rsp,
        stamper,
        gz,
        ros_gz_bridge,
        rviz_node,
        OpaqueFunction(function=launch_setup)
    ])
