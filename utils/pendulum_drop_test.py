import mujoco
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


ANGLE_OFFSET = 180
# Load the Real Hardware Data
try:
    # Read Data
    real_data = pd.read_csv('drop_test_data.csv', header=None, names=['Sample', 'Angle'])
    
    # Reconstruct time: Each sample is 10ms (0.01 seconds)
    real_time = real_data['Sample'] * 0.01 
    
    # Shift real time so the drop starts at exactly t=0
    real_time = real_time - real_time.iloc[0]
    real_angle = real_data['Angle'] + ANGLE_OFFSET
    
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
duration = 15.0  
if model:
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.0         
    data.qpos[1] = (3 * np.pi) / 2.0

    steps = int(duration / model.opt.timestep)
    sim_time = np.zeros(steps)
    sim_angle = np.zeros(steps)

    print(f"Running MuJoCo simulation for {duration} seconds...")

    for i in range(steps):
        
        data.qpos[0] = 0.0
        data.qvel[0] = 0.0
        
        mujoco.mj_step(model, data)
        
        sim_time[i] = data.time
        sim_angle[i] = np.degrees(data.qpos[1])

# Plot Sim vs Real
plt.figure(figsize=(12, 6))

if has_real_data:
    plt.plot(real_time, real_angle, label="Real Hardware (Trimmed CSV)", color='dodgerblue', linewidth=2)

if model:
    plt.plot(sim_time, sim_angle, label="MuJoCo Simulation", color='crimson', linestyle='--', linewidth=2)

plt.title("Sim-to-Real: Pendulum Drop Test Alignment")
plt.xlabel("Time (seconds)")
plt.ylabel("Pendulum Angle (degrees)")
plt.axhline(ANGLE_OFFSET, color='black', linewidth=1, label="Bottom Dead Center")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, duration)
plt.ylim(-100 + ANGLE_OFFSET, 100 + ANGLE_OFFSET)
plt.tight_layout()

print("Close the plot window to exit.")
plt.show()