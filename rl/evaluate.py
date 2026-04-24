import os
import sys
import time
import argparse
import mujoco.viewer

# Ensure Python can find the envs/ module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from envs.furuta_env import FurutaPendulumEnv

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Evaluate a trained Furuta Pendulum PPO model.")
    
    # Argument for the task mode
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["balance", "swing_up"], 
        default="balance",
        help="Choose the task mode: 'balance' or 'swing_up'. Default is 'balance'."
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=None,
        help="Name of the model file (e.g., 'my_model.zip'). Assumes it is located in the ./saved_models/ directory."
    )

    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Full exact path to the model file. Overrides --model_name if provided."
    )
    
    args = parser.parse_args()

    # Determine the model path based on the provided arguments
    if args.model_path:
        model_path = args.model_path
    elif args.model_name:
        # Automatically point to the saved_models directory
        model_path = os.path.join("./saved_models", args.model_name)
        # Automatically append .zip if you forget to type it in the command line
        if not model_path.endswith('.zip'):
            model_path += '.zip'
    else:
        # Fallback to the defaults based on the mode
        if args.mode == "balance":
            model_path = "./saved_models/ppo_furuta_balance_final.zip"
        else:
            model_path = "./saved_models/ppo_furuta_swing_up_final.zip"

    # 1. Load the Environment
    env = FurutaPendulumEnv(mode=args.mode)
    
    # 2. Load the Trained Model
    print(f"--- Initialization ---")
    print(f"Task Mode: {args.mode.upper()}")
    print(f"Loading model from: {model_path}")
    
    try:
        model = PPO.load(model_path, device="cpu")
    except FileNotFoundError:
        print(f"\nError: Could not find model at {model_path}.")
        print("Check the spelling or verify the file exists.")
        sys.exit(1)

    # 3. Reset the environment to get the initial observation
    obs, info = env.reset()

    print("\nLaunching MuJoCo Viewer. Press ESC to exit.")
    
    # 4. Launch the MuJoCo interactive viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        
        while viewer.is_running():
            step_start = time.time()
            
            # A. Ask the Neural Network for the best action based on current sensors
            action, _states = model.predict(obs, deterministic=True)
            
            # B. Step the environment physics forward using that action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # C. Update the visualizer
            viewer.sync()
            
            # D. If it fails (spins out of control/falls), reset it and try again
            if terminated or truncated:
                print("Episode terminated. Resetting...")
                obs, info = env.reset()
                
            # E. Maintain real-time speed so it doesn't look like a blur
            
            physics_dt = env.model.opt.timestep * env.frame_skip
            time_until_next_step = physics_dt - (time.time() - step_start)
        
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()