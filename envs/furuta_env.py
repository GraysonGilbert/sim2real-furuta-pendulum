import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class FurutaPendulumEnv(gym.Env):
    def __init__(self, xml_path="../sim/models/furuta_pendulum.xml", mode="balance"):
        super().__init__()
        
        self.mode = mode
        self.frame_skip = 10
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.max_current = 1.5      # Max Amps for the GM4108
        self.pend_cpr = 16384       # AS5047P encoder resolution
        self.rotor_cpr = 8192       # CUI Devices encoder resolution    
        
        self.pend_resolution = (2 * np.pi) / self.pend_cpr
        self.rotor_resolution = (2 * np.pi) / self.rotor_cpr
        
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        self.prev_current = 0.0
        
        # real world characteristic additions
        self.nom_gear = self.model.actuator_gear.copy()
        self.nom_damping = self.model.dof_damping.copy()
        self.nom_friction = self.model.dof_frictionloss.copy()
        self.nom_mass = self.model.body_mass.copy()

        self.action_delay_buffer = [0.0, 0.0]
        
    def step(self, action):
        # Scale the action to real Amps
        target_current = np.clip(action[0], -1.0, 1.0) * self.max_current
        
        self.action_delay_buffer.append(target_current)
        delayed_current = self.action_delay_buffer.pop(0)
        
        # Apply the current to the actuator
        self.data.ctrl[0] = delayed_current
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # Step the MuJoCo physics
        # mujoco.mj_step(self.model, self.data)
        
        rotor_pos_raw = self.data.sensor('rotor_angle').data[0]
        rotor_vel_raw = self.data.sensor('rotor_vel').data[0]
        pend_pos_raw = self.data.sensor('pendulum_angle').data[0]
        pend_vel_raw = self.data.sensor('pendulum_vel').data[0]
        
        pend_pos_wrapped = ((pend_pos_raw + np.pi) % (2 * np.pi)) - np.pi
        
        raw_state = (rotor_pos_raw, rotor_vel_raw, pend_pos_wrapped, pend_vel_raw)
        
        # Get quantized observations
        obs = self._get_obs(raw_state)
        
        # Calculate reward and termination
        reward = self._get_reward(raw_state, target_current, self.prev_current)
        
        self.prev_current = target_current
        
        terminated, term_reason = self._get_terminated(raw_state)
        
        if terminated and term_reason == "Base_drift":
            reward -= 50.0
        
        truncated = False 
        
        info = {}
        if terminated:
            info["termination_reason"] = term_reason
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        self.model.actuator_gear = self.nom_gear * self.np_random.uniform(0.9, 1.1)
        self.model.dof_damping = self.nom_damping * self.np_random.uniform(0.5, 1.5)
        self.model.dof_frictionloss = self.nom_friction * self.np_random.uniform(0.5, 1.5)
        self.model.body_mass = self.nom_mass * self.np_random.uniform(0.95, 1.05)
        
        self.prev_current = 0.0
        self.consecutive_upright_steps = 0
        
        if self.mode == "balance":
            initial_pend_angle = self.np_random.uniform(-0.15, 0.15)
        elif self.mode == "swing_up":
            # initial_pend_angle = self.np_random.uniform(-np.pi, np.pi)
            initial_pend_angle = np.pi + self.np_random.uniform(-0.1, 0.1)
            
        self.data.qpos[1] = initial_pend_angle
        
        mujoco.mj_forward(self.model, self.data)
        
        rotor_pos_raw = self.data.sensor('rotor_angle').data[0]
        rotor_vel_raw = self.data.sensor('rotor_vel').data[0]
        pend_pos_raw = self.data.sensor('pendulum_angle').data[0]
        pend_vel_raw = self.data.sensor('pendulum_vel').data[0]
        
        pend_pos_wrapped = ((pend_pos_raw + np.pi) % (2 * np.pi)) - np.pi
        
        raw_state = (rotor_pos_raw, rotor_vel_raw, pend_pos_wrapped, pend_vel_raw)
        
        return self._get_obs(raw_state), {}

    def _get_obs(self, raw_state):
        """Extracts, wraps, and quantizes sensor data to [-1.0, 1.0]."""
        
        rotor_pos_raw, rotor_vel_raw, pend_pos_wrapped, pend_vel_raw = raw_state
        
        # Quantize positions
        rotor_pos_quantized = np.round(rotor_pos_raw / self.rotor_resolution) * self.rotor_resolution
        pend_pos_quantized = np.round(pend_pos_wrapped / self.pend_resolution) * self.pend_resolution
        
        rotor_noise_scale = 0.05 + 0.1 * abs(rotor_vel_raw)
        pend_noise_scale = 0.02 + 0.05 * abs(pend_vel_raw)
        
        noisy_rotor_vel = rotor_vel_raw + self.np_random.normal(0, rotor_noise_scale)
        noisy_pend_vel = pend_vel_raw + self.np_random.normal(0, pend_noise_scale)
        
        MAX_ROTOR_POS = 4 * np.pi   # Defined by termination limits
        # MAX_PEND_POS = np.pi        # Max possible value after wrapping
        MAX_ROTOR_VEL = 50.0        # rad/s
        MAX_PEND_VEL = 50.0         # rad/s
        
        obs = np.array([rotor_pos_quantized / MAX_ROTOR_POS,
                        noisy_rotor_vel / MAX_ROTOR_VEL,
                        np.sin(pend_pos_quantized),
                        np.cos(pend_pos_quantized),
                        noisy_pend_vel / MAX_PEND_VEL,
                        self.prev_current / self.max_current],
                        dtype=np.float32)
        
        return np.clip(obs, -1.0, 1.0)


    """
    The current reward for balance works great for genertaing a robust pendulum balancer starting from the top position
    """
    def _get_reward(self, raw_state, current, prev_current):
        rotor_pos, rotor_vel, pend_pos, pend_vel = raw_state
        
        # 0 is Top Dead Center, pi or -pi is bottom position
        upright_reward = np.cos(pend_pos) - 1
        distance_from_perfect_top = abs(pend_pos)
        
        effort_penalty = 0.001 * (current ** 2)
            
        action_rate_penalty = 0.5 * ((current - prev_current) ** 2)
        
        if self.mode == "balance":
            centering_penalty = 0.05 * (rotor_pos ** 2)
            near_top_multiplier = np.clip((np.cos(pend_pos) + 1.0) / 2.0, 0.0, 1.0)
            vel_penalty = near_top_multiplier * (0.1 * (pend_vel ** 2) + 0.05 * (rotor_vel ** 2))
        
            if distance_from_perfect_top < 0.006:
                vel_penalty = 0.0
                action_rate_penalty = 0.0
            
            total_reward = upright_reward - (vel_penalty + centering_penalty + effort_penalty + action_rate_penalty)
            
            return float(total_reward)
        
        elif self.mode == "swing_up":
            centering_penalty = 0.01 * (rotor_pos ** 2)
            
            momentum_bonus = np.clip(0.1 * abs(pend_vel), 0.0, 0.8) # was 1.5
            swing_base_vel_penalty = 0.01 * (rotor_vel ** 2)
            
            catch_vel_penalty = 0.02 * (pend_vel ** 2) + 0.05 * (rotor_vel ** 2)
            
            if distance_from_perfect_top > 0.6:
                catch_weight = 0.0
            elif distance_from_perfect_top < 0.2:
                catch_weight = 1.0
            else:
                catch_weight = (1.2 - distance_from_perfect_top) / 1.0
            
            swing_weight = 1.0 - catch_weight
            
            blended_momentum = momentum_bonus * swing_weight
            blended_vel_penalty = (swing_base_vel_penalty * swing_weight) + (catch_vel_penalty * catch_weight)
            
            if catch_weight == 1.0:
                self.consecutive_upright_steps += 1
                self.catch_bonus = 3.0
            else:
                self.consecutive_upright_steps = 0
                self.catch_bonus = 0.0
                
            duration_bonus = 0.01 * self.consecutive_upright_steps
            
            total_reward = (upright_reward + self.catch_bonus + blended_momentum + duration_bonus) - (blended_vel_penalty + centering_penalty + effort_penalty + action_rate_penalty)
            
            return float(total_reward)

    def _get_terminated(self, raw_state):
        """Ends the episode only for hardware safety reasons."""
        rotor_pos, _, pendulum_pos, _ = raw_state
        
        # HARDWARE SAFETY: Fail if the base spins more than 2 full rotations
        base_out_of_bounds = abs(rotor_pos) > (4 * np.pi)
        
        if self.mode == "balance":
            task_failed = abs(pendulum_pos) > (0.2618)
            
        elif self.mode == "swing_up":
            task_failed = False
            
        termination_reason = None
        if base_out_of_bounds:
            termination_reason = "Base_drift"
        elif task_failed:
            termination_reason = "Pendulum_drop"
        
        return bool(base_out_of_bounds or task_failed), termination_reason

# Self-test loop
if __name__ == "__main__":
    env = FurutaPendulumEnv()
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    print("Environment test completed successfully!")