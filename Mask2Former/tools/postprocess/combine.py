import os
import sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def load_npy_file(file_path):
    return np.load(file_path)

def process_files(npy_files_pred, npy_files_res, folder_path_pred, folder_path_res):
    arrays_by_shape_pred = defaultdict(list)
    arrays_by_shape_res = defaultdict(list)
    
    with ThreadPoolExecutor() as executor:
        futures_pred = {executor.submit(load_npy_file, os.path.join(folder_path_pred, pred_file)): pred_file for pred_file in npy_files_pred}
        futures_res = {executor.submit(load_npy_file, os.path.join(folder_path_res, res_file)): res_file for res_file in npy_files_res}
        
        for future in tqdm(futures_pred, total=len(futures_pred), desc="Loading pred npy files"):
            pred_file = futures_pred[future]
            pred_array = future.result()
            arrays_by_shape_pred[pred_array.shape].append(pred_array)
        
        for future in tqdm(futures_res, total=len(futures_res), desc="Loading res npy files"):
            res_file = futures_res[future]
            res_array = future.result()
            arrays_by_shape_res[res_array.shape].append(res_array)
    
    return arrays_by_shape_pred, arrays_by_shape_res

def save_combined_arrays(arrays_by_shape, file_type, output_file_dir):
    for shape, arrays in arrays_by_shape.items():
        combined_array = np.concatenate([array[np.newaxis, ...] for array in arrays], axis=0)
        shape_str = '_'.join(map(str, shape))
        if file_type == "pred":
            tmp_shape_str = "_".join(shape_str.split("_")[1:])
            output_file = f"{file_type}_{tmp_shape_str}.npy"
        else:
            output_file = f"{file_type}_{shape_str}.npy"
        output_file_path = os.path.join(output_file_dir, output_file)
        np.save(output_file_path, combined_array)
        print(f"Combined {file_type} array of shape {shape} saved to {output_file_path}")

def combine_npy_files(folder_path_pred, folder_path_res, output_file_dir):
    # List all npy files in the folders
    npy_files_pred = sorted([f for f in os.listdir(folder_path_pred) if f.endswith('.npy')])
    npy_files_res = sorted([f for f in os.listdir(folder_path_res) if f.endswith('.npy')])
    
    # Check if the number of files matches
    if len(npy_files_pred) != len(npy_files_res):
        print("The number of pred and res files do not match!")
        return
    
    # Ensure that files are correctly paired
    for pred_file, res_file in zip(npy_files_pred, npy_files_res):
        pred_prefix = os.path.splitext(pred_file)[0].rsplit('_', 1)[0]
        res_prefix = os.path.splitext(res_file)[0].rsplit('_', 1)[0]
        if pred_prefix != res_prefix:
            print(f"Mismatched file pairs: {pred_file} and {res_file}")
            return
    
    arrays_by_shape_pred, arrays_by_shape_res = process_files(npy_files_pred, npy_files_res, folder_path_pred, folder_path_res)
    
    save_combined_arrays(arrays_by_shape_pred, 'pred', output_file_dir)
    save_combined_arrays(arrays_by_shape_res, 'res', output_file_dir)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python combine_npy.py <folder_path_pred> <folder_path_res> <output_file_dir>")
        sys.exit(1)
    
    folder_path_pred = sys.argv[1]
    folder_path_res = sys.argv[2]
    output_file_dir = sys.argv[3]
    combine_npy_files(folder_path_pred, folder_path_res, output_file_dir)
