import os
import sys
import time
import argparse
from typing import Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from envs.furuta_env import FurutaPendulumEnv


LOG_DIR = "../rl/logs/"
os.makedirs(LOG_DIR, exist_ok=True)


# Define the Learning Rate Decay Schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class DeathTrackerCallback(BaseCallback):
    """
    Custom callback for logging the specific causes of episode terminations 
    to TensorBoard without slowing down the training loop.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.base_drift_deaths = 0
        self.pendulum_drop_deaths = 0

    def _on_step(self) -> bool:
    
        for i in range(len(self.locals["dones"])):
            if self.locals["dones"][i]:
                info = self.locals["infos"][i]
                if "termination_reason" in info:
                    if info["termination_reason"] == "Base_drift":
                        self.base_drift_deaths += 1
                    elif info["termination_reason"] == "Pendulum_drop":
                        self.pendulum_drop_deaths += 1
        return True

    def _on_rollout_end(self) -> None:
        # Log the totals to TensorBoard
        self.logger.record("deaths/base_drift", self.base_drift_deaths)
        self.logger.record("deaths/pendulum_drop", self.pendulum_drop_deaths)
        
        # Reset counters for the next rollout
        self.base_drift_deaths = 0
        self.pendulum_drop_deaths = 0


def make_env(mode:str) -> Callable:
    def _init():
        max_episode_steps = 1000
        env = FurutaPendulumEnv(mode=mode)
        env = TimeLimit(env, max_episode_steps)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train the Furuta Pendulum PPO Agent.")
    parser.add_argument( "--mode", 
                        type=str, 
                        choices=["balance", "swing_up"], 
                        default="balance", help="Training phase: 'balance' (Phase 1) or 'swing_up' (Phase 2)."
                        )
    parser.add_argument("--load_model_dir",
                        type=str,
                        default=None,
                        help="Directory of the saved Phase 1 model. REQUIRED when running in swing_up mode."
                        )
    args = parser.parse_args()
    
    epoch_time = str(int(time.time()))
    MODEL_DIR = f"../rl/saved_models/{args.mode}_{epoch_time}/"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"--- Starting Training in {args.mode.upper()} Mode ---")
    
    num_cpu = 10
    
    vec_env = SubprocVecEnv([make_env(args.mode) for _ in range(num_cpu)])
    
    # 100,000 steps across all cpus
    save_freq = 100_000 // num_cpu
    
    checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                             save_path=MODEL_DIR,
                                             name_prefix=f"ppo_furuta_{args.mode}"
                                             )
    
    death_tracker = DeathTrackerCallback()
    callback_list = CallbackList([checkpoint_callback, death_tracker])
    
    if args.mode == "balance":
        print("Initializing new PPO model with random weights...")
        model = PPO("MlpPolicy",
                    vec_env,
                    verbose=1,
                    tensorboard_log=LOG_DIR,
                    learning_rate=linear_schedule(0.0001),
                    n_steps=2048,
                    batch_size=2048,
                    n_epochs=10, 
                    ent_coef=0.005,
                    clip_range=0.2,
                    device="cpu"
                    )
    
    elif args.mode == "swing_up":
        
        print("Initializing new PPO model with random weights...")
        model = PPO("MlpPolicy",
                    vec_env,
                    verbose=1,
                    tensorboard_log=LOG_DIR,
                    learning_rate=linear_schedule(0.0003),
                    n_steps=2048,
                    batch_size=2048,
                    n_epochs=10,
                    ent_coef=0.03,
                    clip_range=0.2,
                    device="cpu"
                    )
        
    
    print(f"Beginning Training on {num_cpu} cores...")
    model.learn(total_timesteps=20_000_000,
                callback=callback_list,
                tb_log_name=f"PPO_{args.mode.capitalize()}_Single_{epoch_time}"
                )
    
    model.save(f"{MODEL_DIR}/ppo_furuta_{args.mode}_final")
    
    print(f"Training Complete. Model saved to {MODEL_DIR}.")
    