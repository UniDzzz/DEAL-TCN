import sys
import os
from tqdm import tqdm  
import pandas as pd
import numpy as np
import re
from scipy.integrate import trapz, simps
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from config import  N_Frame, type_dic

# Get the first command line argument
if len(sys.argv) < 2:
    print("Usage: python lammps_output_process.py <work_path>")
    sys.exit(1)
work_path = sys.argv[1]

bondfile_name = "bonds.reaxff"
dumpfile_name = "atominfo.lmp"

### Data Reading ###
# Read atom count
atom_num_path = os.path.join(work_path, bondfile_name)
with open(atom_num_path, "r") as atom_num_file:
    for _ in range(2):
        atom_num_file.readline()
    n_atom = int(atom_num_file.readline().split()[4])

# Read the atom type corresponding to the atom id
bonds_path = os.path.join(work_path, bondfile_name)
with open(bonds_path, "r") as bonds_file:
    atom_type_dic = {}
    for _ in range(7):  # Skip the first 7 lines
        bonds_file.readline()
    for _ in range(n_atom):
        line = bonds_file.readline().split()
        atom_id = int(line[0])
        atom_type = int(line[1])
        atom_type_dic[atom_id] = atom_type

### Function Definitions ###
# Input atom id lists e.g.[1,3,5,6], output corresponding atom type list [1,2,2,3]
def identify1(input_list):
    output_list = list(map(lambda x: atom_type_dic[x], input_list))
    return output_list
# Input atom type lists e.g.[1,2,2,3], output [Mo,O,O,S] 
def identify2(input_list):
    output_list = list(map(lambda x: type_dic[x],input_list))
    return output_list
# Read the input list [Mo,O,O,S], output molecular formula as MoO2S, elements in the formula are sorted by the element index in type_dic
def identify3(lists):
    type_lists = []
    for i,j in type_dic.items():
        type_lists.append(j)
    str_o = ''
    for i in type_lists:
        if lists.count(i) == 0:
            str_i=''
        elif lists.count(i) == 1:
            str_i = i #Mo
        elif lists.count(i) > 1:
            str_i = i + str(lists.count(i))
        str_o = str_o + str_i
    return str_o

def preprocess_data():
    ### Main Logic ###
    # Build the atom connection dictionary for each frame
    list_N_Frame = [] ####################For module 2 to call
    Time_steps = []
    Timestep_path = os.path.join(work_path, bondfile_name)
    list_N_Frame_path =   os.path.join(work_path, bondfile_name)
    with open(list_N_Frame_path, "r") as bonds_file, open(Timestep_path, "r") as timestep_file:
        for i in tqdm(range(N_Frame), desc="Preprocessing data, please wait patiently"):  # Use tqdm to add a progress bar
            if i == 0:
                pre_line = 7
                timesteps = timestep_file.readline().split()[2]
                for n in range(n_atom+6):
                    timestep_file.readline() # Read and discard the 6 lines of text below the "Timestep 500" line plus n_atom lines of atom information.
            else:
                pre_line = 8
                timestep_file.readline()
                timesteps = timestep_file.readline().split()[2]
                for n in range(n_atom+6):
                    timestep_file.readline()
            Time_steps.append(timesteps)
            
            # Read bonds file
            for _ in range(pre_line):
                bonds_file.readline()
            dic_per_total = {}
            for _ in range(n_atom):
                line = bonds_file.readline().split()
                lists = [int(i) for i in line if '.' not in i]
                del lists[1:3]
                del lists[-1]
                dic_per_total[lists[0]] = lists[1:]
            list_N_Frame.append(dic_per_total)

    # Build a list of molecule information for each frame, including cluster information (all atom ids contained in each molecule); number of molecules; molecular formula, and output the product file.
    List_N_cluster = []  # Store clusters for all frames
    List_N_cluster_Num = []  # Store the number of molecules in all frames
    List_N_Molecular = []  # Store the molecule types for all frames
    Dic_N_cluster = []  # Store the molecular formula information corresponding to each atom id for each frame

    species_out_path = os.path.join(work_path, "species.out")

    # Cluster identification and processing
    with open(species_out_path, "w") as speicies_out:
        for step, frame in enumerate(list_N_Frame):
            timesteps = Time_steps[step]
            visited = set()  # Store processed atoms
            List_per_clusters = []  # Store cluster information for each frame
            dic_per_frame = {}  # Store molecular formula information corresponding to each atom ID in the current frame
            # Depth-first search to identify clusters
            for atom in frame:
                if atom not in visited:
                    molecule = []  # List of atoms in the current cluster
                    List_per_clusters.append(molecule)
                    stack = [atom]

                    while stack:
                        current_atom = stack.pop()
                        if current_atom not in visited:
                            molecule.append(current_atom)
                            visited.add(current_atom)
                            for neighbor in frame[current_atom]:
                                if neighbor not in visited:
                                    stack.append(neighbor)

            # Cluster type identification
            List_per_molecular = [identify3(identify2(identify1(cluster))) for cluster in List_per_clusters]
            # Build mapping from each atom ID to molecular formula
            for cluster, molecular in zip(List_per_clusters, List_per_molecular):
                for atom_id in cluster:
                    dic_per_frame[atom_id] = molecular

            # Update result storage lists
            List_N_cluster.append(List_per_clusters)
            List_N_cluster_Num.append(len(List_per_clusters))
            List_N_Molecular.append(List_per_molecular)
            Dic_N_cluster.append(dic_per_frame)
            
            # Output to species.out file
            Species = {} 
            for product in List_per_molecular:
                Species[product] = Species.get(product,0)+1
            Species_lists = [product for product in Species.keys()]
            Species_counts = [counts for counts in Species.values()]
            Species_counts_str = list(map(lambda x:str(x), Species_counts)) ###Convert numbers to strings for writing to file
            speicies_out.write("#"+" "+"Timestep"+"    "+"No_Specs"+"    "+"No_Cluster"+"    "+" ".join(Species_lists) + "\n")
            speicies_out.write(timesteps +"            " +" " + str(len(Species_lists)) + "            "+str(len(List_per_clusters))+"       " +"  ".join(Species_counts_str) +"\n")   

    return  atom_type_dic,list_N_Frame,List_N_Molecular,Dic_N_cluster

atom_type_dic,list_N_Frame,List_N_Molecular,Dic_N_cluster = preprocess_data()

def sepcies_statistical_analysis(List_N_Molecular):
    def generate_all_molecules_matrix(List_N_Molecular):
        """
        Generate a quantity matrix for all molecule types changing over time.

        Parameters:
        - List_N_Molecular: Nested list of molecule types for each frame.

        Returns:
        - all_molecules_matrix: Quantity change matrix for each molecule type over time.
        - molecules_index: Mapping from molecule type to matrix row index.
        """
        # Identify all molecule species that have appeared
        all_molecules = set()
        for frame in List_N_Molecular:
            all_molecules.update(frame)
        molecules_index = {molecule: i for i, molecule in enumerate(all_molecules)}    # Create an index for each molecule type
        all_molecules_matrix = np.zeros((len(all_molecules), len(List_N_Molecular)))    # Initialize the matrix
        for frame_idx, frame in enumerate(List_N_Molecular):    # Fill the matrix
            for molecule in frame:
                all_molecules_matrix[molecules_index[molecule], frame_idx] += 1

        return all_molecules_matrix, molecules_index

    ###Smooth all_molecules_matrix
    # Use Gaussian smoothing to smooth short-term fluctuations and highlight long-term trends.
    def gaussian_smoothing(matrix, sigma=2):
        smoothed_matrix = np.zeros_like(matrix)
        for row_idx, row in enumerate(matrix):
            smoothed_matrix[row_idx] = gaussian_filter1d(row, sigma=sigma)
        return smoothed_matrix

    # Process all values less than 1 to 0
    def threshold_filter(matrix, threshold):
        # Use NumPy's where function to set data points below the threshold to 0
        filtered_matrix = np.where(matrix < threshold, 0, matrix)
        return filtered_matrix

    ###Define the Species_statistical_info table
    # Define molecule category
    def assign_group(molecule):
        """
        Classify the molecule into types based on its composition of different element combinations.

        Parameters:
        - molecule: Molecular formula string.
        - type_dic: Dictionary of element types, key is element type number, value is element symbol.

        Returns:
        - Classified molecule type string.
        """
        # Generate reverse mapping from element symbol to type number
        symbol_to_type = {v: k for k, v in type_dic.items()}

        # Identify all element types contained in the molecule
        elements = set(re.findall(r'[A-Z][a-z]*', molecule))

        # Generate molecule type string
        molecule_type = []
        for symbol in sorted(elements, key=lambda x: symbol_to_type[x]):  # Sort according to the order in type_dic
            count = len(re.findall(symbol, molecule))
            molecule_type.append(f"{symbol}{count if count > 1 else ''}")

        return ''.join(molecule_type)
    # Define the time centroid of the molecule, integrate the quantity distribution curve over time, and find the time corresponding to half the area
    def compute_time_centroids(matrix):
        centroids = []
        for row in matrix:
            total_quantity = np.sum(row)
            half_quantity = total_quantity / 2
            accumulated_quantity = 0
            centroid_time = 0
            for idx, quantity in enumerate(row):
                accumulated_quantity += quantity
                if accumulated_quantity >= half_quantity:
                    centroid_time = idx
                    break
            centroids.append(centroid_time)
        return centroids
    # Define the molecule quantity integration stability weight, i.e., the area of the quantity vs. time evolution curve
    def compute_quantity_integration_stability(matrix):
        quantity_integration_stability = []
        for row in matrix:
            y = [i for i in row]
            x = np.linspace(1, len(y), len(y))  
            # Use Simpson's rule for integration
            area_simps = simps(y, x)
            quantity_integration_stability.append(area_simps)
        return quantity_integration_stability

    # Define the molecule quantity average stability weight, i.e., the mean of the quantity vs. time evolution curve
    def compute_quantity_average_stability(matrix):
        quantity_average_stability = []
        for row in matrix:
            a = sum(row)
            b= np.count_nonzero(row)
            if b == 0:
                quantity_average_stability.append(0)
            else:
                quantity_average_stability.append(a/b)
        return quantity_average_stability

    # Create Species_statistical_info table
    def Species_statistical_info_df(matrix):
        df = pd.DataFrame({
            'Species_name':list(all_molecule_indices.keys()),
            'Integration_stability':compute_quantity_integration_stability(matrix),
            'Average_stability': compute_quantity_average_stability(matrix),
            'Time Centroid':compute_time_centroids(matrix),
            'Group': [assign_group(i) for i in list(all_molecule_indices.keys())]
        })

        return df
    # Create a function to filter Species_statistical_info, select molecules with stability ratio within a set threshold, this function is optional.
    def filter_top_percent_within_group(df, top_n_percent):
        """
        For each "Group", sort the molecules in that Group by Integration_stability weight in descending order,
        keep the top top_n_percent number of molecules, rounding down.

        Parameters:
        - df: DataFrame containing molecule statistics info.
        - top_n_percent: Percentage of top molecules to keep in each group, value range 0 to 1.

        Returns:
        - Filtered DataFrame.
        """
        def filter_group(group_df):
            # Calculate the number of molecules to keep in each group, round down, set to zero
            n_to_keep = int(len(group_df) * top_n_percent)
            # If n_to_keep is 0 but there is data in the group, keep at least one molecule
            #n_to_keep = max(n_to_keep, 1)
            # Sort by Integration_stability descending and keep the top n_to_keep molecules
            return group_df.sort_values(by="Integration_stability", ascending=False).head(n_to_keep)

        # Group the DataFrame by 'Group' and apply the filtering rule
        filtered_df = df.groupby('Group').apply(filter_group).reset_index(drop=True)
        return filtered_df

    # Use the function to process List_N_Molecular
    all_molecules_matrix, all_molecule_indices = generate_all_molecules_matrix(List_N_Molecular)############################################
    gs_mo_matrix = gaussian_smoothing(all_molecules_matrix, sigma=2)#################
    all_molecules_matrix_smooth = threshold_filter(gs_mo_matrix, threshold=1)########################################################
    # Generate Species_statistical_info_df, sort it, and output Species_statistical_info.xlsx file
    info_path = os.path.join(work_path, "Species_statistical_info.xlsx")
    info_filter_path = os.path.join(work_path, "Species_statistical_info_filter.xlsx")
    
    species_statistical_info_df = Species_statistical_info_df(all_molecules_matrix_smooth)
    species_statistical_info_df = species_statistical_info_df.sort_values(by=["Group", "Average_stability"], ascending=[True, False]).reset_index(drop=True)
    species_statistical_info_df_filter = species_statistical_info_df[species_statistical_info_df['Average_stability'] > 2.11]
    species_statistical_info_df_filter = filter_top_percent_within_group(species_statistical_info_df_filter,1) # This df stores the filtered information of all species, including all elements
    if not os.path.exists(info_path):
        species_statistical_info_df.to_excel(info_path, index=False)
    else:
        print(f"{info_path} already exists, skipping write.")

    if not os.path.exists(info_filter_path):
        species_statistical_info_df_filter.to_excel(info_filter_path, index=False)
    else:
        print(f"{info_filter_path} already exists, skipping write.")

    return species_statistical_info_df, species_statistical_info_df_filter

## 3.1 Output state tracking file for target element: state_trace_data.csv (noise molecule removed version), reaction network analysis is based on this file
### Process Dic_N_cluster, function input element_type = 1, this value is the number corresponding to the element in type_dic
element_type = 1 # The number corresponding to the element in type_dic

def capture_target_atom_track_notclean(element_type,atom_type_dic,Dic_N_cluster):

    def capture_target_atom_id(element_type,atom_type_dic):
        Goal_atom_list = []
        for x,y in atom_type_dic.items():
            if y == element_type:
                Goal_atom_list.append(x)
        return Goal_atom_list
    target_atom_id_list = capture_target_atom_id(element_type,atom_type_dic)

    # Check if the file exists
    if os.path.exists(os.path.join(work_path,'state_trace_data.csv')):
        print("Hello, state_trace_data.csv have existed")
    else:
        total_frame_result = [] # Used to store all intermediate states of all target atoms, this list will contain len(target_atom_id_list) small lists, each small list contains N_Frame molecular formulas.
        for atom in tqdm(target_atom_id_list, desc="File is being written, please wait patiently"):
        #for atom in target_atom_id_list: # Iterate over each target atom
            result=[] # Create an empty list
            for i in range(len(Dic_N_cluster)): # Iterate over each frame
                molecular = Dic_N_cluster[i][int(atom)]
                result.append(molecular)
            total_frame_result.append(result)

        total_frame_result_df = pd.DataFrame(total_frame_result)
        total_frame_result_df.to_csv(os.path.join(work_path,'state_trace_data.csv'),index=False)

def capture_target_atom_track_clean(element_type,atom_type_dic,Dic_N_cluster,List_N_Molecular):# Filter out noise molecules in Spcies_name_noise, i.e., if a state is a noise molecule, this state takes the value of the previous state. If the first state is a noise molecule, the first state can be filled with this noise molecule. Allow noise to enter the table
    a,b = sepcies_statistical_analysis(List_N_Molecular)
    Spcies_name_filter = b['Species_name'].tolist()

    def capture_target_atom_id(element_type,atom_type_dic):
        Goal_atom_list = []
        for x,y in atom_type_dic.items():
            if y == element_type:
                Goal_atom_list.append(x)
        return Goal_atom_list
    target_atom_id_list = capture_target_atom_id(element_type,atom_type_dic)
    # Check if the file exists
    if os.path.exists(os.path.join(work_path,'state_trace_data_clean.csv')):
        print("Hello, state_trace_data_clean.csv have existed")
    else:
        total_frame_result = [] 
        for atom in tqdm(target_atom_id_list, desc="File is being written, please wait patiently"):
        #for atom in target_atom_id_list: # Iterate over each target atom
            result=[] # Create an empty list
            previous_molecular = None
            for frame in Dic_N_cluster: # Iterate over each frame
                molecular = frame[int(atom)]
                if molecular in Spcies_name_filter:
                    result.append(molecular)
                    previous_molecular = molecular
                else: ### This implementation allows rows with noise molecules to exist, and subsequent noise molecules may appear in that row; if it doesn't start with a noise molecule, the entire row will not contain noise molecules.
                    if previous_molecular is None:
                        result.append(molecular)
                    else:
                        result.append(previous_molecular)
            total_frame_result.append(result)
        
        total_frame_result_df = pd.DataFrame(total_frame_result)
        total_frame_result_df.to_csv(os.path.join(work_path,'state_trace_data_clean.csv'),index=False)

def capture_target_atom_track_filter(element_type,atom_type_dic,Dic_N_cluster,List_N_Molecular):# Filter out noise molecules in Spcies_name_noise, i.e., if a state is a noise molecule, this state takes the value of the previous state. If the first state is a noise molecule, the first state can be filled with this noise molecule. Allow noise to enter the table
    a,b = sepcies_statistical_analysis(List_N_Molecular)
    Spcies_name_filter = b['Species_name'].tolist()

    def capture_target_atom_id(element_type,atom_type_dic):
        Goal_atom_list = []
        for x,y in atom_type_dic.items():
            if y == element_type:
                Goal_atom_list.append(x)
        return Goal_atom_list
    target_atom_id_list = capture_target_atom_id(element_type,atom_type_dic)
    # Check if the file exists
    if os.path.exists(os.path.join(work_path,'state_trace_data_filter.csv')):
        print("Hello, state_trace_data_filter.csv have existed")
    else:
        total_frame_result = [] 
        for atom in tqdm(target_atom_id_list, desc="File is being written, please wait patiently"):
            result = []  # Create an empty list
            previous_molecular = None
            first_frame = True  # Add a variable to mark if it is the first frame
            for frame in Dic_N_cluster:  # Iterate over each frame
                molecular = frame[int(atom)]
                if molecular in Spcies_name_filter:
                    result.append(molecular)
                    previous_molecular = molecular
                    first_frame = False  # Once the first frame is processed, set this variable to False
                else:
                    if first_frame:  # If it's the first frame and a noise molecule
                        break  # Break the loop directly, no longer track this atom
                    else:
                        if previous_molecular is not None:
                            result.append(previous_molecular)
                        # If the first state is noise, it won't execute here, because it already broke in the 'if first_frame' above
            if not first_frame:  # If at least one frame was processed (i.e., this atom was not completely skipped)
                total_frame_result.append(result)
        total_frame_result_df = pd.DataFrame(total_frame_result)
        total_frame_result_df.to_csv(os.path.join(work_path,'state_trace_data_filter.csv'),index=False)

element_type = 1
capture_target_atom_track_filter(element_type,atom_type_dic,Dic_N_cluster,List_N_Molecular)

input_file = os.path.join(work_path,'state_trace_data_filter.csv')
output_file = os.path.join(work_path,'clean_state_trace_data_filter.csv')

# Adjust the sep parameter according to the actual situation (e.g., sep=',' if comma-separated)
# header=None means there is no row in the file that can be used as column names, read all as data
df = pd.read_csv(input_file, sep=',', header=None)

# The first row of df is the frame number, subsequent rows are the molecular states
# No need for frame info column names, do not write the first row to the new table, only process subsequent rows
processed_rows = []

# Iterate through each row starting from the second row (index 1) of the data for processing
for idx, row in df.iloc[1:].iterrows():
    # row is a Series, where values are all molecular states of the current row (30000 columns)
    states = row.values  
    # Convert it to a modifiable list
    updated_states = list(states)
    
    # Start from the second molecular state (index 1) up to the second-to-last molecular state (index len-2)
    for i in range(1, len(updated_states) - 1):
        # If the preceding and succeeding molecular states are the same, and different from the current state, update the current state to be consistent with the preceding and succeeding ones !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Please read
        if updated_states[i-1] == updated_states[i+1] and updated_states[i] != updated_states[i-1]:
            updated_states[i] = updated_states[i-1]
    
    # Save the processed results to the list
    processed_rows.append(updated_states)

# Convert the processed results to a DataFrame and write to a csv file
processed_df = pd.DataFrame(processed_rows)
processed_df.to_csv(output_file, sep=',', header=False, index=False)

####4. Continue processing clean_state_trace_data_filter.csv data and obtain species_time_matrix_initial.npy################
# Fix code for the last part
if os.path.exists(output_file):
    try:
        # Read CSV file
        df = pd.read_csv(output_file)
        # Skip the first row (column names), start processing from the second row
        data_rows = df.iloc[1:].values
        
        print(f"Data dimensions: {data_rows.shape}")
        
        # Get all unique molecular states
        all_states = set()
        for row in data_rows:
            for state in row:
                if pd.notna(state):  # Exclude NaN values
                    all_states.add(str(state).strip())
        
        species_list = sorted(list(all_states))
        num_species = len(species_list)
        num_frames = data_rows.shape[1]  # Should be 30000
        
        print(f"Found {num_species} different molecular states")
        print(f"Number of time frames: {num_frames}")
        
        # Create mapping from species to index
        species_to_idx = {species: idx for idx, species in enumerate(species_list)}
        
        # Initialize result matrix (number of species × number of time frames)
        species_time_matrix = np.zeros((num_species, num_frames), dtype=int)
        
        # Count the number of each molecular state at each time frame
        for frame_idx in range(num_frames):
            if frame_idx % 5000 == 0:
                print(f"Processing progress: {frame_idx}/{num_frames}")
            
            # Get the states of all atoms at the current time frame
            frame_states = []
            for row_idx in range(data_rows.shape[0]):
                state = data_rows[row_idx, frame_idx]
                if pd.notna(state):
                    frame_states.append(str(state).strip())
            
            # Count the number of each state in the current frame
            state_counts = Counter(frame_states)
            
            # Update the matrix
            for species, count in state_counts.items():
                if species in species_to_idx:
                    species_time_matrix[species_to_idx[species], frame_idx] = count
        
        # Save results to the same folder
        output_dir = os.path.dirname(output_file)
        
        # Save species_time_matrix as .npy file
        matrix_file = os.path.join(output_dir, 'species_time_matrix_initial.npy')
        np.save(matrix_file, species_time_matrix)
        
        # Also save the species list for future use
        species_file = os.path.join(output_dir, 'species_list_initial.txt')
        with open(species_file, 'w') as f:
            for species in species_list:
                f.write(f"{species}\n")
        
        print(f"Successfully processed and saved files")
        print(f"Matrix dimensions: {species_time_matrix.shape}")
        print(f"Save location: {matrix_file}")
        
    except Exception as e:
        print(f"Error processing file {output_file}: {e}")
else:
    print(f"File not found: {output_file}")