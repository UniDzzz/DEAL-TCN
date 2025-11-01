#!/usr/bin/env python
"""
Initialize Parameter Pool Script
Used to generate 500 parameter points and run initial LAMMPS simulations
"""

import os
import sys
import numpy as np
import shutil
import subprocess
from pyDOE import lhs  # Latin Hypercube Sampling
import json
from datetime import datetime

# ============ Configuration Parameters ============
TOTAL_POOL_SIZE = 500
INITIAL_FRAMES = 500  # N1 - Initial simulation frames
RANDOM_SEED = 42  # Fixed random seed for reproducibility

# Parameter Ranges
T_MIN, T_MAX = 1100, 1500  # Temperature range (K)
P_MIN, P_MAX = 2, 6        # Pressure range (atm)
RATIO_MIN, RATIO_MAX = 1/25, 1/15  # Raw material ratio range

# Path Configuration
BASE_DIR = '/dssg/home/acct-umjwym/umjwym-zzd/AL_GasKit_V1.0'
POOL_DATA_DIR = os.path.join(BASE_DIR, 'Pool_data')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'lammps_input_file')
WORK_PATH = BASE_DIR

# Create necessary directories
os.makedirs(POOL_DATA_DIR, exist_ok=True)

def generate_lhs_points(n_samples, seed=42):
    """
    Use Latin Hypercube Sampling to generate parameter points
    
    Args:
        n_samples: Number of sample points
        seed: Random seed
    
    Returns:
        points: (n_samples, 3) array, containing (T, P, ratio)
    """
    np.random.seed(seed)
    
    # Generate LHS samples in the [0,1] range
    lhs_samples = lhs(3, samples=n_samples)
    
    # Map to actual parameter ranges
    points = np.zeros((n_samples, 3))
    points[:, 0] = T_MIN + (T_MAX - T_MIN) * lhs_samples[:, 0]  # Temperature
    points[:, 1] = P_MIN + (P_MAX - P_MIN) * lhs_samples[:, 1]  # Pressure
    points[:, 2] = RATIO_MIN + (RATIO_MAX - RATIO_MIN) * lhs_samples[:, 2]  # Ratio
    
    return points

def save_pool_points(points, output_file='pool_points.txt'):
    """
    Save parameter points to a text file
    
    Args:
        points: Parameter points array
        output_file: Output filename
    """
    filepath = os.path.join(BASE_DIR, output_file)
    with open(filepath, 'w') as f:
        f.write("# Temperature(K)\tPressure(atm)\tRatio(Mo3O9/S2)\n")
        for i, (T, P, ratio) in enumerate(points):
            f.write(f"{i+1}\t{T:.2f}\t{P:.2f}\t{ratio:.6f}\n")
    print(f"Parameter points saved to: {filepath}")
    
    # Also save in JSON format for easy subsequent reading
    pool_dict = {
        'points': points.tolist(),
        'metadata': {
            'total_size': len(points),
            'T_range': [T_MIN, T_MAX],
            'P_range': [P_MIN, P_MAX],
            'ratio_range': [RATIO_MIN, RATIO_MAX],
            'random_seed': RANDOM_SEED,
            'created_time': datetime.now().isoformat()
        }
    }
    json_file = os.path.join(BASE_DIR, 'pool_points.json')
    with open(json_file, 'w') as f:
        json.dump(pool_dict, f, indent=2)
    print(f"JSON format saved to: {json_file}")

def create_simulation_folder(T, P, ratio, parent_dir):
    """
    Create simulation folder and copy necessary files
    
    Args:
        T: Temperature
        P: Pressure
        ratio: Raw material ratio
        parent_dir: Parent directory path
    
    Returns:
        sim_dir: Simulation directory path
    """
    # Construct folder name
    denom = 1 / ratio
    denom_str = f"{denom:.0f}" if abs(denom - round(denom)) < 1e-6 else f"{denom:.2f}"
    P_str = f"{P:.0f}" if abs(P - round(P)) < 1e-6 else f"{P:.1f}"
    folder_name = f"{int(T)}K_{P_str}atm_1per{denom_str}"
    
    sim_dir = os.path.join(parent_dir, folder_name)
    os.makedirs(sim_dir, exist_ok=True)
    
    # Copy template files
    template_files = [
        "in.MoO3S",
        "ffield.reax.Mo_Al_O_S",
        "Mo3O9.dat",
        "S2.dat",
        "lammps_siyuan.slurm"
    ]
    
    for file in template_files:
        src = os.path.join(TEMPLATE_DIR, file)
        dst = os.path.join(sim_dir, file)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: Template file {src} does not exist")
    
    return sim_dir

def modify_lammps_input(sim_dir, T, P, ratio, frames):
    """
    Modify LAMMPS input file parameters
    
    Args:
        sim_dir: Simulation directory
        T: Temperature
        P: Pressure
        ratio: Raw material ratio
        frames: Simulation frames
    """
    in_path = os.path.join(sim_dir, "in.MoO3S")
    
    # Calculate parameters
    denom = 1 / ratio
    Mo3O9_cnt = 150
    S2_cnt = round(Mo3O9_cnt * denom)
    
    # Calculate box size
    kB = 1.380649e-23
    box_size = round((((Mo3O9_cnt + S2_cnt) * kB * T) / (P * 101325)) ** (1/3) * 1e10)
    
    # Calculate total steps
    steps_per_frame = 5000
    run_steps = steps_per_frame * int(frames)
    
    # Read and modify file
    new_lines = []
    with open(in_path, 'r') as f:
        for line in f:
            if line.strip().startswith("velocity all create"):
                parts = line.split()
                parts[3] = f"{float(T):.1f}"
                line = " ".join(parts) + "\n"
            elif " nvt " in line and " temp " in line:
                parts = line.split()
                parts[5] = parts[6] = str(int(T))
                line = " ".join(parts) + "\n"
            elif "create_atoms" in line and " mol " in line:
                parts = line.split()
                if "mol" in parts:
                    idx = parts.index("mol") + 1
                    if idx < len(parts) and parts[idx] == "m2":
                        parts[3] = str(S2_cnt)
                        line = " ".join(parts) + "\n"
            elif line.strip().startswith("region") and " block " in line:
                parts = line.split()
                parts[4] = parts[6] = parts[8] = str(box_size)
                line = " ".join(parts) + "\n"
            elif line.strip().startswith("run"):
                line = f"run {run_steps}\n"
            
            new_lines.append(line)
    
    # Write back to file
    with open(in_path, 'w') as f:
        f.writelines(new_lines)

def run_lammps_simulation(sim_dir):
    """
    Run LAMMPS simulation
    
    Args:
        sim_dir: Simulation directory
    
    Returns:
        success: Whether successfully completed
    """
    in_path = os.path.join(sim_dir, "in.MoO3S")
    
    # Build command
    cmd = (
        "module load lammps/20230328-intel-2021.4.0-omp && "
        f"mpirun -np 48 lmp -in {in_path}"
    )
    
    print(f"Starting simulation: {os.path.basename(sim_dir)}")
    
    try:
        # Run simulation
        proc = subprocess.Popen(
            cmd,
            cwd=sim_dir,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Monitor output
        count = 0
        done = False
        for line in proc.stdout:
            count += 1
            if count % 100 == 0:
                print(".", end="", flush=True)
            if "Total wall time:" in line:
                print(f"\nCompleted: {line.strip()}")
                done = True
                break
        
        proc.wait()
        
        if not done:
            print(f"\nWarning: {sim_dir} simulation may not have completed normally")
            return False
            
        # Run post-processing script
        proc_script = os.path.join(WORK_PATH, "lammps_output_process.py")
        if os.path.exists(proc_script):
            ret = subprocess.run(
                ["python", proc_script, sim_dir],
                capture_output=True,
                text=True
            )
            if ret.returncode != 0:
                print(f"Post-processing failed: {ret.stderr}")
                return False
        else:
            print(f"Warning: Post-processing script {proc_script} does not exist")
        
        return True
        
    except Exception as e:
        print(f"Simulation error: {e}")
        return False

def check_simulation_complete(sim_dir):
    """
    Check if simulation is complete
    
    Args:
        sim_dir: Simulation directory
    
    Returns:
        bool: Whether complete
    """
    log_file = os.path.join(sim_dir, "log.lammps")
    if not os.path.exists(log_file):
        return False
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        if lines:
            last_line = lines[-1].strip()
            return "Total wall time:" in last_line
    
    return False

def main():
    """Main function"""
    print("="*60)
    print("Active Learning Framework - Parameter Pool Initialization")
    print("="*60)
    
    # Step 1: Generate parameter points
    print("\nStep 1: Generating parameter pool points...")
    points = generate_lhs_points(TOTAL_POOL_SIZE, seed=RANDOM_SEED)
    print(f"Generated {TOTAL_POOL_SIZE} parameter points")
    
    # Save parameter points
    save_pool_points(points)
    
    # Step 2: Create simulation folders and prepare input files
    print("\nStep 2: Preparing simulation folders...")
    pool_info = []
    
    for i, (T, P, ratio) in enumerate(points):
        print(f"\rPreparing point {i+1}/{TOTAL_POOL_SIZE}...", end="")
        
        # Create folder
        sim_dir = create_simulation_folder(T, P, ratio, POOL_DATA_DIR)
        
        # Modify input file
        modify_lammps_input(sim_dir, T, P, ratio, INITIAL_FRAMES)
        
        # Record information
        pool_info.append({
            'index': i,
            'T': T,
            'P': P,
            'ratio': ratio,
            'sim_dir': sim_dir,
            'status': 'prepared'
        })
    
    print("\nAll simulation folders prepared")
    
    # Save pool information
    pool_info_file = os.path.join(BASE_DIR, 'pool_info.json')
    with open(pool_info_file, 'w') as f:
        json.dump(pool_info, f, indent=2)
    print(f"Pool information saved to: {pool_info_file}")
    
    # Ask whether to start running simulations
    print("\n" + "="*60)
    print("Initialization preparation complete!")
    print(f"A total of {TOTAL_POOL_SIZE} parameter points have been prepared")
    print(f"Each point will run an initial simulation of {INITIAL_FRAMES} frames")
    print("\nNote: Running all simulations will take a significant amount of time")
    
    response = input("\nStart running all simulations immediately? (y/n): ")
    
    if response.lower() == 'y':
        print("\nStarting to run all simulations...")
        successful = 0
        failed = []
        
        for i, info in enumerate(pool_info):
            print(f"\n[{i+1}/{TOTAL_POOL_SIZE}] Running simulation: {os.path.basename(info['sim_dir'])}")
            
            success = run_lammps_simulation(info['sim_dir'])
            
            if success:
                successful += 1
                info['status'] = 'completed'
            else:
                failed.append(info['sim_dir'])
                info['status'] = 'failed'
            
            # Update status file
            with open(pool_info_file, 'w') as f:
                json.dump(pool_info, f, indent=2)
        
        print("\n" + "="*60)
        print(f"Simulation completion statistics:")
        print(f"Successful: {successful}/{TOTAL_POOL_SIZE}")
        print(f"Failed: {len(failed)}")
        
        if failed:
            print("\nFailed simulations:")
            for dir in failed:
                print(f"   - {os.path.basename(dir)}")
    else:
        print("\nSimulation preparation complete, you can run them manually later")
        print("Use the run_pool_simulations.py script to run simulations in batch")
    
    print("\nInitialization script execution finished!")

if __name__ == "__main__":
    main()