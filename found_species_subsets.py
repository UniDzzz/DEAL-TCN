import os
import numpy as np
import re
import sys

# All incoming paths, each argument after the script name
paths = sys.argv[1:]
# Example: Print all received paths
print("Received species list files:", paths)
print("Starting to find common species from all files and perform group statistics...")

# If you want to delete certain major groups, write their names here; set to None or [] when not deleting
# Example: Delete Mo_group and MoO_group
del_species = ['Mo_group', 'MoO_group','MoOS_group','MoS_group']  # or None, or []

# Unify into list form
if del_species is None:
    del_list = []
elif isinstance(del_species, str):
    del_list = [del_species]
else:
    del_list = del_species

# Part 1: Collect species sets from all files (read _initial files)
all_species_sets = []
valid_dirs = []
for i, file_path in enumerate(paths):
    dir_path = os.path.dirname(file_path)
    list_initial = os.path.join(dir_path, 'species_list_initial.txt')
    mat_initial  = os.path.join(dir_path, 'species_time_matrix_initial.npy')

    if not os.path.exists(list_initial) or not os.path.exists(mat_initial):
        print(f"Warning: Folder {dir_path} is missing species_list_initial.txt or species_time_matrix_initial.npy, skipping")
        continue

    with open(list_initial, 'r') as f:
        species_list = [line.strip() for line in f if line.strip()]
    all_species_sets.append(set(species_list))
    valid_dirs.append(dir_path)
    print(f"Folder {i+1}: {dir_path}, initial species count: {len(species_list)}")

if not all_species_sets:
    raise RuntimeError("No valid initial species list files were found")

# Find intersection: common species
common_species = set.intersection(*all_species_sets)
common_species_list = sorted(common_species)
print(f"Common species count: {len(common_species_list)}")
print("Common species:", common_species_list)

# Define grouping function
def get_group(species_name):
    elems = set(re.findall(r'([A-Z][a-z]?)', species_name))
    key = elems.intersection({'Mo', 'O', 'S'})
    if   key == {'Mo'}:         return 'Mo_group'
    elif key == {'Mo','O'}:     return 'MoO_group'
    elif key == {'Mo','S'}:     return 'MoS_group'
    elif key == {'Mo','O','S'}: return 'MoOS_group'
    else:                       return None

# Fixed order of the four major groups
all_groups = ['Mo_group', 'MoO_group', 'MoS_group', 'MoOS_group']

# Part 2: Process each folder—read from _initial, write to standard files after processing
for i, dir_path in enumerate(valid_dirs):
    print(f"\nProcessing folder {i+1}/{len(valid_dirs)}: {dir_path}")
    list_initial = os.path.join(dir_path, 'species_list_initial.txt')
    mat_initial  = os.path.join(dir_path, 'species_time_matrix_initial.npy')
    list_out     = os.path.join(dir_path, 'species_list.txt')
    mat_out      = os.path.join(dir_path, 'species_time_matrix.npy')

    # Read initial species list and matrix
    with open(list_initial, 'r') as f:
        original_species = [line.strip() for line in f if line.strip()]
    original_matrix = np.load(mat_initial)
    n_frames = original_matrix.shape[1]

    # Keep common species
    common_idx = [original_species.index(sp) for sp in common_species_list]
    filtered = original_matrix[common_idx, :]

    # Initialize summary vectors and member lists for all major groups
    group_sums    = {g: np.zeros(n_frames, dtype=int) for g in all_groups}
    group_members = {g: [] for g in all_groups}

    # Collect all non-common species and accumulate by group
    other_species = [sp for sp in original_species if sp not in common_species]
    for sp in other_species:
        grp = get_group(sp)
        if grp is None:
            print(f"   Unrecognized group: {sp}, skipping")
        else:
            idx = original_species.index(sp)
            group_sums[grp]    += original_matrix[idx, :]
            group_members[grp].append(sp)

    # Print non-common species included in each major group
    print("   Non-common species group statistics:")
    for g in all_groups:
        members = group_members[g]
        print(f"     {g}: {len(members)} items -> {members}")

    # Determine the final groups to write: remove del_list from all_groups
    invalid = [g for g in del_list if g not in all_groups]
    if invalid:
        print(f"   Warning: The following categories to be deleted are invalid and will be ignored: {invalid}")
    output_groups = [g for g in all_groups if g not in del_list]
    if del_list:
        print(f"   Deleted categories: {[g for g in del_list if g in all_groups]}")

    # Construct new matrix: common species + remaining groups
    extras = [group_sums[g] for g in output_groups]
    new_mat = np.vstack([filtered] + extras)

    # Write to output files
    np.save(mat_out, new_mat)
    with open(list_out, 'w') as f:
        # First write common species
        for sp in common_species_list:
            f.write(sp + '\n')
        # Then write the retained major group names, maintaining order
        for g in output_groups:
            f.write(g + '\n')

    print(f"   Update complete: Saved to {mat_out}, new matrix dimensions {new_mat.shape}")

print("\nAll files updated: species_time_matrix.npy and species_list.txt have been generated.")