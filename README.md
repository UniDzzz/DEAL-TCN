Based on the provided repository files, here is a comprehensive README in English.

***

# Deep Ensemble Active Learning for Chemical Simulation Prediction

## Overview

This repository contains the source code for an Active Learning (AL) framework designed to accelerate the exploration of chemical reaction parameter spaces. The framework uses a Deep Ensemble of Temporal Convolutional Networks (TCNs) to model and predict the time-evolution of chemical species from LAMMPS molecular dynamics simulations.

The primary goal is to efficiently model a chemical system (specifically the reaction of MoO3 and S2) under various conditions (Temperature, Pressure, Ratio). Instead of brute-forcing the entire parameter space, the AL framework intelligently selects the most informative simulations to run. It does this by quantifying the ensemble's prediction uncertainty (variance) and prioritizing new simulations where the models disagree the most.

This repository includes scripts for parameter pool generation, simulation post-processing, data alignment, the core active learning loop, and benchmarking against random sampling and other deep learning models.

## Core Components

* **Parameter Space Initialization**: Generates an initial pool of 500 parameter combinations (Temperature, Pressure, Ratio) using Latin Hypercube Sampling (LHS).
* **LAMMPS Simulation**: Runs ReaxFF molecular dynamics simulations using the provided LAMMPS templates (`lammps_input_file/`).
* **Data Post-Processing**: Parses raw LAMMPS bond output (`bonds.reaxff`) to identify molecular clusters at each timestep. It then generates species concentration time-series matrices (`species_time_matrix_initial.npy`).
* **Species Alignment**: A utility (`found_species_subsets.py`) finds the common subset of species across all simulations and groups rare species (e.g., `Mo_group`, `MoO_group`) to ensure a consistent input shape for the models.
* **Deep Ensemble TCN**: The core predictive model is an ensemble of Temporal Convolutional Networks. Model diversity is enhanced using **Bagging** (training each model on a different bootstrap sample of the data) and **hyperparameter randomization** (using a different dropout rate for each model).
* **Active Learning Loop**: The main script (`Main_code.ipynb`) iteratively trains the ensemble, calculates uncertainty on all un-trained points in the pool, selects the `TOP_K` most uncertain points, and adds them to the training set for the next iteration.
* **Benchmarking**: The notebook also includes a framework for comparing the performance of the final AL-trained model against a model trained on a randomly selected dataset of the same size. It also includes baseline comparisons against other architectures like LSTM, GRU, RNN, BiLSTM, and Transformer.

## Key Files

* `Main_code.ipynb`: The main Jupyter Notebook containing the active learning loop, model definitions (TCN, LSTM, GRU, etc.), random sampling comparison, and visualization code.
* `initialize_pool.py`: Script to generate the initial 500-point parameter pool using LHS, create simulation directories, and run the initial simulations.
* `lammps_output_process.py`: Post-processes LAMMPS output (`bonds.reaxff`) into species-vs-time matrices (`.npy`) and species lists (`.txt`).
* `found_species_subsets.py`: Utility to align species data across multiple simulations by finding a common subset and grouping the rest.
* `config.py`: Basic configuration defining the number of frames (`N_Frame`) and the atom type mapping (`type_dic`).
* `lammps_input_file/`: Directory containing templates for LAMMPS simulations, including:
    * `in.MoO3S`: The main LAMMPS input script.
    * `ffield.reax.Mo_Al_O_S`: The ReaxFF force field file.
    * `Mo3O9.dat`, `S2.dat`: Molecule data files for reactants.
    * `lammps_siyuan.slurm`: A SLURM batch script for running LAMMPS on a cluster.

## Workflow

1.  **Initialize Pool**: Run `initialize_pool.py`. This script will:
    * Generate a `pool_info.json` file mapping parameters to simulation directories.
    * Create 500 subdirectories in `Pool_data/`, each populated with modified LAMMPS input files.
    * Offer to run all 500 initial simulations (for `INITIAL_FRAMES`).
2.  **Run Simulations**: Allow the initialization script to run the simulations, or run them manually (e.g., by submitting the `.slurm` script in each directory).
    * The `lammps_output_process.py` script is called automatically after each simulation to generate the `species_time_matrix_initial.npy` and `species_list_initial.txt` files.
3.  **Run Active Learning**: Open and execute the `Main_code.ipynb` notebook.
    * It will first select an initial small training set (e.g., `INITIAL_TRAIN_SIZE = 1`).
    * It then enters the main loop:
        1.  Aligns species data using `found_species_subsets.py`.
        2.  Trains the TCN ensemble on the current training set.
        3.  Evaluates the ensemble's performance on the test set.
        4.  Calculates uncertainty (prediction variance) for all remaining points in the pool.
        5.  Selects the `TOP_K` points with the highest uncertainty and adds them to the training set.
        6.  Repeats for `N_ITERATIONS`.
4.  **Benchmark & Visualize**: The notebook contains further cells to:
    * Run the random sampling comparison test.
    * Train and evaluate other baseline models (LSTM, GRU, Transformer).
    * Generate plots comparing the parameter space coverage and MSE evolution of AL vs. Random Sampling.

## Requirements

* Python 3.x
* PyTorch
* NumPy
* pandas
* scikit-learn
* pyDOE (for LHS)
* tqdm
* matplotlib
* numba (Optional, for DTW metric optimization)
* A working installation of LAMMPS (for data generation).
