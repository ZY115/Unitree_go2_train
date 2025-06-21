# Introduction
This repository originates from https://github.com/Zhefan-Xu/isaac-go2-ros2.git. For the convenience of team collaboration, the modified code has been temporarily uploaded to GitHub.

## Current progress:
Successfully deployed all environments and reproduced the original repository

Successfully obtained Lidar and depth camera information

Successfully transmitted depth camera information to obs

## Issues to be resolved:
The input dimensions of the model have been changed from 235 → 307 435, and we need a new strategy for training

Since the Lidar we are using, Hesai_XT32_SD10, is not part of the IsaacSim default devices, we need to find a new method to incorporate Lidar information

We need to find a method that can combine the basic information from Go2, Lidar information, and depth camera information for training

### If using the same environment settings as this repository, provide the following example startup code:

Path settings
```bash
export PYTHONPATH=$PYTHONPATH:$HOME/Desktop/IsaacLab1.2/IsaacLab-1.2.0/source/extensions/omni.isaac.lab

export PYTHONPATH=$PYTHONPATH:$HOME/Desktop/IsaacLab1.2/IsaacLab-1.2.0/source/extensions/omni.isaac.lab_tasks

export PYTHONPATH=$PYTHONPATH:$HOME/Desktop/IsaacLab1.2/IsaacLab-1.2.0/source/extensions/omni.isaac.lab_assets
```

Run

./python.sh ~/Desktop/isaac-go2-ros2/isaac_go2_ros2.py

### Please modify according to your actual path
