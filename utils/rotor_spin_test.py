import mujoco
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load the Real Hardware Data 
try:
    # Read the CSV data
    real_data = pd.read_csv('rotor_test_data.csv')
    
    # Convert Time_ms to seconds and reset sample time to 0
    real_time = (real_data['Time_ms'] - 11130)/ 1000.0 
    real_vel = real_data['Velocity_rads']
    
    has_real_data = True
    print("Successfully loaded real hardware data!")
except Exception as e:
    print(f"Could not load real data: {e}")
    has_real_data = False

# Load the MuJoCo Model
xml_path = "../models/furuta_pendulum.xml" 

if not os.path.exists(xml_path):
    print(f"WARNING: Could not find XML at {xml_path}")
    model = None
else:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

# Run Simulation 
duration = 7 
if model:
    mujoco.mj_resetData(model, data)
    
    # Capture the pendulum's initial "hanging" position so we can lock it there
    locked_pendulum_pos = data.qpos[1] 

    steps = int(duration / model.opt.timestep)
    sim_time = np.zeros(steps)
    sim_vel = np.zeros(steps)

    print(f"Running MuJoCo simulation for {duration} seconds...")
    
    integral_error = 0.0
    
    for i in range(steps):
        t = data.time
        
        # Lock the Pendulum at 0
        data.qpos[1] = locked_pendulum_pos
        data.qvel[1] = 0.0 
            
        if t < 5.0:
            target_vel = 20.0
            error = target_vel - data.qvel[0]
            
            integral_error += error * model.opt.timestep
            
            # PI Gains for simulated motor response
            p_term = 0.01 * error         
            i_term = 0.125 * integral_error
            
            total_effort = p_term + i_term
            
            data.ctrl[0] = np.clip(total_effort, -3.0, 3.0)
            
        else:
            # Power is cut, motor freewheels
            data.ctrl[0] = 0.0 
            
        mujoco.mj_step(model, data)
        
        sim_time[i] = t
        sim_vel[i] = data.qvel[0]

# Plot Sim vs Real
plt.figure(figsize=(12, 6))

if has_real_data:
    plt.plot(real_time, real_vel, label="Real Hardware (ESP32 CSV)", color='dodgerblue', linewidth=2)

if model:
    plt.plot(sim_time, sim_vel, label="MuJoCo Simulation", color='crimson', linestyle='--', linewidth=2)

plt.title("Sim-to-Real: Motor Coast-Down Test (Rotor Friction)")
plt.xlabel("Time (seconds)")
plt.ylabel("Base Velocity (rad/s)")
plt.axhline(0, color='black', linewidth=1)
plt.axvline(10, color='gray', linestyle=':', label='Power Cut (Coast Start)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-1.0, duration)
plt.tight_layout()

print("Close the plot window to exit.")
plt.show()