[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# **Robust Control of a Furuta Pendulum: Leveraging PPO and MuJoCo for Sim2Real Transfer**

**ENPM690 Final Project – University of Maryland**

**Team Members:**
* Grayson Gilbert

---
## Table of Contents

1. [Overview](#overview)
2. [Project Sewtup](#project-setup)
3. [Repository Layout](#repository-layout)
4. [Project Workflow](#project-workflow)
   - [Train PPO Model](#1-train-ppo-model)
   - [Evaluate Model](#2-evaluate-model)
   - [Export Policy to Header File](#3-export-policy-to-header-file)
   - [Flash Microcontroller to Run Policy on Hardware](#4-flash-microcontroller-to-run-policy-on-hardware)
5. [Hardware Components](#hardware-components)
5. [License](#license)


## **Overview**
This project aims to bridge the gap between high-fidelity physics based simulation and physical hardware by developing a robust Proximal Policy Optimization (PPO) based controller for a furuta pendulum. The project trains an agent that can repeatedly swing up and balance a pendulum, while handling the noise, friction, and non-linearities of the real world. This was achieved through simulation using MuJoCo before being transferred to a physial furuta pendulum.

---

## **Repository Layout**

```
sim2real-furuta_pendulum/
├── demos/                     # Project demonstration pictures and videos
├── envs/                      # Custom Gymnasium training environment
├── hardware/                  # Arduino sketch to control ESP32 and Solidworks assembly
├── rl/
│   ├── exported_models/       # Exported PPO model weight header files
│   ├── logs/                  # Model training Tensorboard logs
│   ├── saved_models/          # Automatically saved PPO models
├── sim/                    
│   ├── hw_sim_debug/          # Model and hardware related debug csv files
│   ├── meshes/                # Pendulum CAD meshes
│   ├── models/                # Pendulum MuJoCo model
├── utils/                     # All python scripts necessary for model viewing, training, evaluation, and debug
├── README.md                  # This file
├── requirements.txt           # Prerequistes
├── Dockerfile                 # Builds project docker container    
```

---
## **Project Setup**

### **Setup via Docker Container**

Download the ```Dockerfile``` from this repository and navigate to its saved location. Then run the following commands:

**Note: The docker container can be built with Nvidia GPU acceleration if the host machine has the available Nvidia hardware. This is the default configuration. If the host machine does not have the required Nvidia GPU, the Dockerfile can be modified for CPU processing only. Follow instructions inside the Dockerfile to modify the file for CPU only processing.**

#### GPU Accelerated Docker Container Setup
```shell
# Build Dockerfile image with 
docker build -t furuta-sim-env .
```

```shell
# Nvidia GPU accelerated docker container
#
# Create docker container from image [Deletes container after exit]
xhost +local:docker
docker run -it --rm \
    --gpus all \
    --device /dev/dri:/dev/dri \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -e MUJOCO_GL=glfw \
    -e __NV_PRIME_RENDER_OFFLOAD=1 \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    furuta-sim-env bash

# Nvidia GPU accelerated docker container
#
# Create docker container from image [Keeps container after exit]
xhost +local:docker
docker run -it \
    --name=mujoco-container
    --gpus all \
    --device /dev/dri:/dev/dri \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -e MUJOCO_GL=glfw \
    -e __NV_PRIME_RENDER_OFFLOAD=1 \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    furuta-sim-env bash
```

#### CPU Only Docker Container Setup:
```shell
# Buid dockerfile with
docker build -t furuta-sim-env-cpu-only .
```
```shell
### CPU only mode
xhost +local:docker
docker run -it --rm \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -e MUJOCO_GL=glfw \
    furuta-sim-env-cpu /bin/bash
```

### **Setup via Cloning Repo**
```shell
# Clone the repository
git clone https://github.com/GraysonGilbert/sim2real-furuta-pendulum.git
```
---


## **Project Workflow**

### **1. Train PPO Model**
* The first step in the project workflow is training the PPO model to swing up and balance the pendulum. Run the following ```train.py``` script to launch the multi-core cpu based model training.

```shell
# From the sim2real-furuta-pendulum directory navigate to utils/
cd utils/
```

```shell
# Kick off training for PPO model
# 
# NOTE: The training script was designed around the Intel(R) Core(TM) Ultra 7 155H CPU.
#       It is reccomended to spread across 10 cpus cores if available. Default option
#       is set to single core training.
#
# Use --help for information about the various CLI arguments

python3 train.py --mode swing_up --num_cpus 10 # Note 10 cores specified

```

```shell
# View Tensorboard logs in a browser while training a model
# In a separate terminal (if running in container, terminal must be in a container as well) run the following
# from the sim2real-furuta-pendulum/ directory

tensorboard --logdir ./rl/logs/
```

### **2. Evaluate Model**

* Evaluate the results of the trained model in the MuJoCo simulation using the following ```evaluate.py``` script. This will the launch a single instance of the MuJoCo model viewer, controlled by a specified model.
```shell
# Evaluate trained policy on MuJoCo model
#
# Default values will run example simualtion on a pre-trained swing up policy
#
# Use --help for information about the various CLI arguments

 python3 evaluate.py --mode swing_up --model_name <Model name saved under /saved_models>
 
 # Example:  python3 evaluate.py --mode swing_up --model_name /swing_up_1777060502/ppo_furuta_swing_up_14000000_steps.zip
```

### **3. Export Policy to Header File**

* The next step will be to convert the resulting trained policy into source code capable of running on memory constrained devices such as an esp32. Use the following command to run a script that converts the trained PPO policy into source code.
```shell
# Export the trained policy weight into a C++ header file for running on embedded devices
#
# NOTE: The model_name argument assumes the model is within the /rl/saved_models/ direcotry
#
# Use --help for information about the various CLI arguments

python3 export_to_cpp.py --model_name <model_name_goes_here> --output <policy_name_goes_here> # Default filename is policy_net.h
```

### **4. Flash Microcontroller to Run Policy on Hardware**
* **Note: Configuring and setting up the esp32 device for use with the Arduino IDE is outside the scope of this project.**
* Move file into your local Arduino directory where libraries are stored. This will allow the ```policy_net.h``` file to be discoverable when compiling the arduino sketch. Copy over ```esp32_PPO_furuta_balancer.ino``` into an Arduino IDE for flashing onto the esp32.

---

## **Hardware Components**

* iPower GM4108h-120T Brushless Gimbal Motor
* CUI Devices AMT103V ABI Encoder
* I2C Bi-directional 3.3V-5V Level Shifter
* ESP32 Lolin Lite Development Board
* DENG FOC V3 Brushless Driver Board
* AS5047P SPI Magnetic Rotary Encoder
* 3D Printed Pendulum Parts
* 304 Stainless Steel Rods for Pendulum Mass


The full Solidworks assembly is available under the ```hardware/``` directory.

---

## **License**

This project is licensed under the **MIT License**.
