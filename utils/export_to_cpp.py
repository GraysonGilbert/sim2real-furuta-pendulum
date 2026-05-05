import argparse
import os
import torch
import numpy as np
from stable_baselines3 import PPO


def export_weights_to_cpp(model_path, header_filename):
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device="cpu")
    
    # Extract the state dict from the PyTorch policy
    state_dict = model.policy.state_dict()
       
    # Extract and transpose weights for easier C++ array indexing
    w1 = state_dict['mlp_extractor.policy_net.0.weight'].numpy().T
    b1 = state_dict['mlp_extractor.policy_net.0.bias'].numpy()
    
    w2 = state_dict['mlp_extractor.policy_net.2.weight'].numpy().T
    b2 = state_dict['mlp_extractor.policy_net.2.bias'].numpy()
    
    w_out = state_dict['action_net.weight'].numpy().T
    b_out = state_dict['action_net.bias'].numpy()

    # Create the C++ Header File
    with open(header_filename, 'w') as f:
        f.write("// Auto-generated PPO Policy Weights\n")
        f.write("#pragma once\n\n")
        f.write("#include <math.h>\n\n")
        
        # Write array dimensions
        f.write(f"const int IN_FEATURES = {w1.shape[0]};\n")
        f.write(f"const int L1_FEATURES = {w1.shape[1]};\n")
        f.write(f"const int L2_FEATURES = {w2.shape[1]};\n")
        f.write(f"const int OUT_FEATURES = {w_out.shape[1]};\n\n")
        
        def write_array(name, arr, is_1d=False):
            if is_1d:
                f.write(f"const float {name}[{arr.shape[0]}] = {{\n")
                f.write(", ".join([f"{x}f" for x in arr]))
                f.write("\n};\n\n")
            else:
                f.write(f"const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n")
                for row in arr:
                    f.write("  {" + ", ".join([f"{x}f" for x in row]) + "},\n")
                f.write("};\n\n")

        write_array("WEIGHT1", w1)
        write_array("BIAS1", b1, True)
        write_array("WEIGHT2", w2)
        write_array("BIAS2", b2, True)
        write_array("W_OUT", w_out)
        write_array("B_OUT", b_out, True)

        # Write the Inference Function
        f.write("""
// Feedforward Neural Network Inference
// Takes a 6-element observation array, returns a 1-element action array
inline void compute_action(const float* obs, float* action) {
    float h1[L1_FEATURES] = {0};
    float h2[L2_FEATURES] = {0};

    // Layer 1 (Linear + Tanh)
    for (int j = 0; j < L1_FEATURES; ++j) {
        float sum = BIAS1[j];
        for (int i = 0; i < IN_FEATURES; ++i) {
            sum += obs[i] * WEIGHT1[i][j];
        }
        h1[j] = tanhf(sum); // Tanh activation
    }

    // Layer 2 (Linear + Tanh)
    for (int j = 0; j < L2_FEATURES; ++j) {
        float sum = BIAS2[j];
        for (int i = 0; i < L1_FEATURES; ++i) {
            sum += h1[i] * WEIGHT2[i][j];
        }
        h2[j] = tanhf(sum); // Tanh activation
    }

    // Output Layer (Linear, No Activation)
    for (int j = 0; j < OUT_FEATURES; ++j) {
        float sum = B_OUT[j];
        for (int i = 0; i < L2_FEATURES; ++i) {
            sum += h2[i] * W_OUT[i][j];
        }
        action[j] = sum; 
    }
}
""")
    print(f"Successfully generated {header_filename}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Stable Basleine3 PPO weights to a C++ header file.")
    
    # Model Argument
    parser.add_argument(
        "-m", "--model_name", 
        type=str, 
        required=True, 
        help="Model filename or path relative to the '../rl/saved_models/' directory (e.g., 'swing_up_1777338929/ppo_furuta_swing_up_final.zip')"
    )
    
    # Output Argument
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="policy_net.h", 
        help="Output header filename to be saved in '../rl/exported_models/' (default: 'policy_net.h')"
    )
    
    args = parser.parse_args()

    # Define base directories based on the project structure
    MODEL_DIR = "../rl/saved_models/"
    EXPORT_DIR = "../rl/exported_models/"
    
    # Ensure the export directory exists
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # Construct absolute or resolved paths
    full_model_path = os.path.join(MODEL_DIR, args.model_name)
    full_output_path = os.path.join(EXPORT_DIR, args.output)
    
    # Run the exporter
    export_weights_to_cpp(full_model_path, full_output_path)