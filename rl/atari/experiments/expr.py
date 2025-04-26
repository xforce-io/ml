#!/usr/bin/env python3

import yaml
import os
import shutil
import subprocess
import itertools
from functools import reduce # Required for nested dictionary access
from datetime import datetime # Import datetime

# --- Helper Functions --- 

def read_config(file_path):
    """Reads a YAML config file."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        print(f"Warning: Config file {file_path} not found. Returning empty config.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}. Returning empty config.")
        return {}

def write_config(file_path, config):
    """Writes a dictionary to a YAML config file."""
    try:
        with open(file_path, 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False)
    except Exception as e:
        print(f"Error writing config file {file_path}: {e}")

def set_nested_value(d, keys, value):
    """Sets a value in a nested dictionary using a dot-separated key string."""
    keys_list = keys.split('.')
    nested_dict = d
    for key in keys_list[:-1]:
        if key not in nested_dict or not isinstance(nested_dict[key], dict):
            nested_dict[key] = {}
        nested_dict = nested_dict[key]
    nested_dict[keys_list[-1]] = value

def ensure_dir_exists(dir_path):
    """Ensures a directory exists, creating it if necessary."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"Created missing directory: {dir_path}")
        except Exception as e:
            print(f"Error creating directory {dir_path}: {e}")
            return False
    return True

def move_artifact(source_path, dest_dir):
    """Moves a file or directory artifact to the destination directory."""
    if not os.path.exists(source_path):
        print(f"Artifact {source_path} not found, skipping move.")
        return
    
    dest_path = os.path.join(dest_dir, os.path.basename(source_path))
    try:
        shutil.move(source_path, dest_path)
        print(f"Moved {source_path} to {dest_path}")
    except Exception as e:
        print(f"Error moving {source_path} to {dest_path}: {e}")

# --- Main Experiment Logic --- 

def run_experiments():
    std_file = 'config/global_std.yaml'
    config_file = 'config/global.yaml'
    results_base_dir = 'experiments/results' 
    
    # Define directories required before run and artifacts to move after run
    required_dirs_before_run = ['log', 'saved_models', 'videos']
    artifacts_to_move = {
        config_file: 'global.yaml', # Source: Destination filename
        'log': 'log',              # Source: Destination dirname
        'saved_models': 'saved_models',
        'videos': 'videos'
    }

    # Define parameter grid
    param_grid = {
        'general.ENV_NAME': ['BreakoutNoFrameskip-v4'],
        'dqn.REPLAY_BUFFER_CAPACITY': [200000, 500000],
        'dqn.LEARNING_RATE': [0.0001, 0.00005],
        'dqn.BATCH_SIZE': [32, 64],
        'dqn.TARGET_UPDATE_FREQUENCY': [5000, 10000],
        'dqn.EPSILON_DECAY_STEPS': [1000000, 500000],
        'dqn.LEARNING_STARTS': [10000, 50000],
    }

    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    total_experiments = len(all_combinations)
    current_experiment = 0

    # Ensure base results directory exists
    ensure_dir_exists(results_base_dir)

    # Backup standard config if it exists
    std_config_exists = os.path.exists(std_file)
    if std_config_exists:
        shutil.copyfile(std_file, config_file)
    else:
        print(f"Warning: Standard config file {std_file} not found. Cannot backup or restore.")

    try:
        for combination in all_combinations:
            current_experiment += 1
            print(f'\n运行实验 {current_experiment}/{total_experiments}')

            # --- Pre-run setup ---
            # Load base config
            config = read_config(std_file) if std_config_exists else {}
            
            # Update config with current combination
            experiment_desc_parts = []
            current_env_name = "UnknownEnv"
            for i, param_name in enumerate(param_names):
                value = combination[i]
                set_nested_value(config, param_name, value)
                if param_name == 'general.ENV_NAME':
                     current_env_name = value
                experiment_desc_parts.append(f"{param_name}={value}")
            experiment_desc = ", ".join(experiment_desc_parts)
            print(f"Parameters: {experiment_desc}")
            
            # Write current config
            write_config(config_file, config)

            # Ensure required directories exist
            print("Ensuring required directories exist...")
            all_dirs_ok = True
            for dir_path in required_dirs_before_run:
                if not ensure_dir_exists(dir_path):
                    all_dirs_ok = False
                    break # Stop if a directory cannot be created
            if not all_dirs_ok:
                print("Skipping experiment due to directory creation error.")
                continue
            
            # --- Run experiment ---
            print(f"Starting experiment run...")
            try:
                subprocess.run(['bash', 'bin/run.sh'], check=True) # Added check=True
                print(f"Experiment run finished successfully.")
            except subprocess.CalledProcessError as e:
                 print(f"Experiment run failed with error code {e.returncode}.")
                 # Decide whether to continue or stop all experiments on failure
                 # continue 
            except Exception as e:
                 print(f"An unexpected error occurred during experiment run: {e}")
                 # continue

            # --- Post-run processing ---
            print(f"Starting post-experiment processing...")
            # 1. Create results directory
            current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S') # Added seconds for uniqueness
            result_folder_name = f"{current_env_name}_{current_time_str}"
            result_dir = os.path.join(results_base_dir, result_folder_name)
            if not ensure_dir_exists(result_dir):
                print("Skipping artifact moving due to result directory creation error.")
                continue
            print(f"Results will be stored in: {result_dir}")

            # 2. Move artifacts
            for source, dest_name in artifacts_to_move.items():
                 # Special handling for config file to use dest_name
                 if source == config_file:
                      dest_path = os.path.join(result_dir, dest_name)
                      if os.path.exists(source):
                          try:
                              shutil.move(source, dest_path)
                              print(f"Moved {source} to {dest_path}")
                          except Exception as e:
                              print(f"Error moving {source} to {dest_path}: {e}")
                      else:
                           print(f"Config file {source} not found, cannot move.")
                 else:
                      move_artifact(source, result_dir)
            
            print(f"Post-experiment processing finished.")

    finally:
        print('\n所有实验已完成。')
        # Restore standard config if it existed
        if std_config_exists:
            if not os.path.exists(config_file):
                 print(f"Config file {config_file} was moved. Restoring from {std_file}.")
            shutil.copyfile(std_file, config_file)
            print(f"Restored {config_file} from {std_file}.")
        else:
            if os.path.exists(config_file):
                 print(f"Standard config {std_file} not found. Cleanup might be needed for {config_file}.")
                 # Consider removing the last generated config file if no standard exists
                 # os.remove(config_file) 

if __name__ == '__main__':
    run_experiments() 