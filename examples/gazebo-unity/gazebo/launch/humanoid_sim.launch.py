import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.conditions import IfCondition
import xacro


def generate_launch_description():
    # Package locations
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_robot_description = get_package_share_directory('robot_description')
    pkg_gazebo_unity_examples = get_package_share_directory('gazebo_unity_examples')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default='humanoid_lab.sdf')
    verbose = LaunchConfiguration('verbose', default='false')
    gui = LaunchConfiguration('gui', default='true')

    # Path to the humanoid robot URDF
    robot_xacro_path = PathJoinSubstitution([pkg_robot_description, 'urdf', 'humanoid.urdf.xacro'])

    # Process the Xacro file to generate the robot description
    robot_description_content = xacro.process_file(
        os.path.join(pkg_robot_description, 'urdf', 'humanoid.urdf.xacro')
    ).toxml()

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': PathJoinSubstitution([get_package_share_directory('gazebo_unity_examples'), 'worlds', world]),
            'verbose': verbose,
            'gui': gui
        }.items()
    )

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description_content
        }]
    )

    # Joint State Publisher node (for simulation)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Spawn the humanoid robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', '/robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0'  # Start slightly above ground to avoid collision on spawn
        ],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Timer to delay spawning until Gazebo is ready
    delayed_spawn_entity = TimerAction(
        period=5.0,  # Wait 5 seconds before spawning
        actions=[spawn_entity]
    )

    # Launch a basic controller manager for the humanoid
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            PathJoinSubstitution([pkg_robot_description, 'config', 'humanoid_controllers.yaml']),
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Load and start joint state broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Load and start humanoid base controller
    humanoid_base_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['humanoid_base_controller'],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Example sensor processing node
    sensor_processor = Node(
        package='gazebo_unity_examples',
        executable='sensor_processor',
        name='sensor_processor',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # RViz2 for visualization (optional)
    rviz = Node(
        condition=IfCondition(gui),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_robot_description, 'rviz', 'humanoid_view.rviz'])],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='humanoid_lab.sdf',
                             description='Choose one of the world files from `/gazebo_unity_examples/worlds`'),
        DeclareLaunchArgument('verbose', default_value='false',
                             description='Set to true to enable verbose Gazebo output'),
        DeclareLaunchArgument('gui', default_value='true',
                             description='Set to false to run Gazebo without GUI'),

        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        delayed_spawn_entity,
        controller_manager,
        joint_state_broadcaster_spawner,
        humanoid_base_controller_spawner,
        sensor_processor,
        rviz,
    ])