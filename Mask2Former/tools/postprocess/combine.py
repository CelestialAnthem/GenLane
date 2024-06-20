import os
import sys
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def combine_npy_files(folder_path, output_file_dir, output_file_prefix):
    # List all npy files in the folder
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    # Dictionary to store lists of arrays by their shape
    arrays_by_shape = defaultdict(list)
    
    # Dictionary to store the mapping of combined files
    combine_mapping = defaultdict(list)
    
    # Load each npy file and append to the corresponding shape list
    for npy_file in tqdm(npy_files, desc="Loading npy files"):
        file_path = os.path.join(folder_path, npy_file)
        array = np.load(file_path)
        arrays_by_shape[array.shape].append((npy_file, array))
    
    # Process each shape group
    for shape, arrays in arrays_by_shape.items():
        # Add a new axis to each array before concatenation
        expanded_arrays = [array[np.newaxis, ...] for _, array in arrays]
        combined_array = np.concatenate(expanded_arrays, axis=0)
        
        shape_str = '_'.join(map(str, shape)) 
        shape_str = shape_str if len(shape_str.split("_")) == 2 else '_'.join(shape_str.split("_")[1:])
        output_file = f"{output_file_prefix}_{shape_str}.npy"
        output_file_path = os.path.join(output_file_dir, output_file)
        
        # Save the combined array to a new npy file
        np.save(output_file_path, combined_array)
        
        # Record the mapping
        combine_mapping[output_file] = [npy_file for npy_file, _ in arrays]
        print(combine_mapping)
        
        # Remove the original npy files
        for npy_file, _ in tqdm(arrays, desc=f"Removing files for shape {shape_str}"):
            file_path = os.path.join(folder_path, npy_file)
            print(file_path)
            if file_path.endswith(".npy"):
                os.remove(file_path)
        
        print(f"Combined array of shape {shape} saved to {output_file_path}")
        print(f"Removed {len(arrays)} original npy files")
    
    # Save the mapping to a JSON file
    # mapping_file_path = os.path.join(folder_path, f"{output_file_prefix}_combine_mapping.json")
    # with open(mapping_file_path, 'w') as mapping_file:
    #     json.dump(combine_mapping, mapping_file, indent=4)
    
    # print(f"Combine mapping saved to {mapping_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python combine_npy.py <folder_path> <output_file_prefix>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    output_file_dir = sys.argv[2]
    output_file_prefix = sys.argv[3]
    combine_npy_files(folder_path, output_file_dir, output_file_prefix)
