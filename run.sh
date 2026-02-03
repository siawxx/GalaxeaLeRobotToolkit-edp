#!/bin/bash
dataset_name=debug
input_dir=/home/user/workspace/data/mcap/
output_dir=/home/user/workspace/data/lerobot/
robot_type=R1Pro # options: R1Pro, R1Lite

export SAVE_VIDEO=1 
export USE_H264=0
export USE_COMPRESSION=0
export IS_COMPUTE_EPISODE_STATS_IMAGE=1
export MAX_PROCESSES=2
export USE_ROS1=0
export USE_TRANSLATION=0

python -m dataset_converter \
    --input_dir /home/wxx/GalaxeaLeRobotToolkit-main/20260116 \
    --output_dir /home/wxx/GalaxeaLeRobotToolkit-main/robot_data \
    --robot_type R1Pro \
    --dataset_name pick_cube