import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class FurutaPendulumEnv(gym.Env):
    def __init__(self, xml_path="../sim/models/furuta_pendulum.xml", mode="balance"):
        super().__init__()
        
        self.mode = mode
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # self.max_current = 1.5  # Max Amps for the GM4108
        self.max_current = 1.5  # Max Amps for the GM4108
        self.cpr = 2048 * 4  # CUI devices encoder settings
        self.encoder_resolution = (2 * np.pi) / self.cpr
        
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def step(self, action):
        # Scale the action to real Amps
        target_current = np.clip(action[0], -1.0, 1.0) * self.max_current
        
        # Apply the current to the actuator
        self.data.ctrl[0] = target_current
        
        # Step the MuJoCo physics
        mujoco.mj_step(self.model, self.data)
        
        # Get quantized observations
        obs = self._get_obs()
        
        # Calculate reward and termination
        reward = self._get_reward(obs, target_current)
        terminated = self._get_terminated(obs)
        truncated = False 
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        
        if self.mode == "balance":
            initial_pend_angle = np.pi + np.random.uniform(-0.05, 0.05)
        elif self.mode == "swing_up":
            initial_pend_angle = 0.0 + np.random.uniform(-0.1, 0.1)
            
        self.data.qpos[1] = initial_pend_angle
        
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Extracts, wraps, and quantizes sensor data to [-1.0, 1.0]."""
        rotor_pos_raw = self.data.sensor('rotor_angle').data[0]
        rotor_vel_raw = self.data.sensor('rotor_vel').data[0]
        pend_pos_raw = self.data.sensor('pendulum_angle').data[0]
        pend_vel_raw = self.data.sensor('pendulum_vel').data[0]
        
        pend_pos_wrapped = ((pend_pos_raw + np.pi) % (2 * np.pi)) - np.pi
        
        # Quantize positions
        rotor_pos_quantized = np.round(rotor_pos_raw / self.encoder_resolution) * self.encoder_resolution
        pend_pos_quantized = np.round(pend_pos_raw / self.encoder_resolution) * self.encoder_resolution
        
        
        MAX_ROTOR_POS = 4 * np.pi   # Defined by termination limits
        MAX_PEND_POS = np.pi        # Max possible value after wrapping
        MAX_ROTOR_VEL = 50.0        # rad/s
        MAX_PEND_VEL = 50.0         # rad/s
        
        obs = np.array([rotor_pos_quantized / MAX_ROTOR_POS,
                        rotor_vel_raw / MAX_ROTOR_VEL,
                        pend_pos_quantized / MAX_PEND_POS,
                        pend_vel_raw / MAX_PEND_VEL],
                        dtype=np.float32)
        
        return np.clip(obs, -1.0, 1.0)

    def _get_reward(self, obs, current):
        rotor_pos, rotor_vel, pend_pos, pend_vel = obs
        distance_to_top = np.arccos(-np.cos(pend_pos))
        
        if self.mode == "balance":
            alive_bonus = 1.0
            theta_penalty = 1.0 * (distance_to_top ** 2)
            vel_penalty = 0.1 * (pend_vel **2) + 0.05 * (rotor_vel ** 2)
            centering_penalty = 0.1 * (rotor_pos ** 2)
            effort_penalty = 0.001 * (current ** 2)
            
            return float(alive_bonus - (theta_penalty + vel_penalty + centering_penalty + effort_penalty))
        
        elif self.mode == "swing_up":
            
            return 0.0

    def _get_terminated(self, obs):
        """Ends the episode only for hardware safety reasons."""
        rotor_pos, pendulum_pos, _, _ = obs
        
        # STRICT HARDWARE SAFETY: Fail if the base spins more than 2 full rotations
        base_out_of_bounds = abs(rotor_pos) > (4 * np.pi) 
        
        if self.mode == "balance":
            task_failed = abs(pendulum_pos) > 0.2618
            
        elif self.mode == "swing_up":
            task_failed = False
        
        return bool(base_out_of_bounds or task_failed)

# Quick self-test loop
if __name__ == "__main__":
    env = FurutaPendulumEnv()
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    print("Environment test completed successfully!")