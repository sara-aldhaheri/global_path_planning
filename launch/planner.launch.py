#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    # Get package directory
    pkg_share = FindPackageShare('global_path_planning').find('global_path_planning')
    
    # params_file_path = os.path.join(
    #     get_package_share_directory('global_path_planning'),
    #     'config',
    #     'gpmp2_planner_config.yaml',
    # )

    # Configuration file path
    config_file = PathJoinSubstitution([
        FindPackageShare('global_path_planning'),
        'config',
        'gpmp2_planner_config.yaml'
    ])
    
    # RViz configuration file path
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('global_path_planning'),
        'rviz',
        'gpmp2_planner.rviz'
    ])

    from launch.conditions import IfCondition
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
            description='Launch RViz for visualization'
        ),
        Node(
            package='global_path_planning',
            executable='global_path_planner',
            name='global_path_planning',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                config_file
            ],
            remappings=[
                # For fast_usv maybe?
            ]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file],
            condition=IfCondition(LaunchConfiguration('launch_rviz'))
        )
    ])