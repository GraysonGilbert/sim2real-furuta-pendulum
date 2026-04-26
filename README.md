[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# **Robust Control of Furuta Pendulum: Leveraging PPO and MuJoCo for Sim2Real Transfer**

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
   - [Evaluate Model`)](#2-evaluate-model)
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
├── envs/
│   ├── mars_overseer/         # Map fusion and global SLAM node
│   ├── mars_fleet_bringup/    # Launch files, configuration, multi-robot simulation
│   ├── mars_exploration/      # Sector-based and frontier-based exploration
├── hardware/                  # Arduino sketch to control ESP32
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

### **Setup via Cloning Repo**

---


## **Project Workflow**

### **1. Train PPO Model**

```shell
# From the sim2real-furuta-pendulum directory navigate to utils/
cd utils/
```

```shell
# Kick off training for PPO model
# 
# NOTE: The training script was designed around the Intel(R) Core(TM) Ultra 7 155H CPU.
#       Please be aware the training will be spread across 10 cpus cores.
#
# Use --help for information about various CLI arguments

python3 train.py --mode swing_up 
```

### **2. Evaluate Model**

*
```shell
 python3 evaluate.py --mode swing_up --model_name <Model name saved under /saved_models>
 
 # Example:  python3 evaluate.py --mode swing_up --model_name /swing_up_1777060502/ppo_furuta_swing_up_14000000_steps.zip

```

### **3. Export Policy to Header File**

* 
```shell
python3 export_to_cpp.py # Modify filepath in python script before running
```

### **4. Flash Microcontroller to Run Policy on Hardware**

* Move file into your local Arduino directory where libraries are stored. This will allow the ```policy_net.h``` file to be discoverable when compiling the arduino sketch.

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

---

## **License**

This project is licensed under the **MIT License**.
