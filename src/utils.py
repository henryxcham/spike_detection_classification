import numpy as np
import os

def load_and_concatenate_npy(file_paths: list) -> np.ndarray:
    """
    Loads data from multiple .npy files, concatenates them, and handles errors.
    """
    data_list = []
    print("Loading data...")
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                print(f"Warning: File not found at '{file_path}'. Skipping.")
                continue
            data = np.load(file_path)
            data_list.append(data)
            print(f"Successfully loaded file: {file_path} with shape {data.shape}")
        except Exception as e:
            print(f"An error occurred while loading '{file_path}': {e}")
            
    if data_list:
        try:
            combined_data = np.concatenate(data_list, axis=0)
            print(f"\nAll arrays concatenated. Final shape: {combined_data.shape}")
            return combined_data
        except ValueError as ve:
            print(f"Error during concatenation: {ve}")
            print("This can happen if the arrays have incompatible shapes.")
            return np.array([])
    else:
        print("\nNo valid files were loaded. Returning an empty array.")
        return np.array([])

def normalize_data(data, mean, std):
    """Standardizes data using a pre-calculated mean and standard deviation."""
    return (data - mean) / std