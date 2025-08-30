import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_power_spectrum(data: np.ndarray, sample_rate: float, title: str, color: str):
    """
    Calculates and plots the power spectrum of a signal.
    """
    num_samples = len(data)
    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(num_samples, 1 / sample_rate)
    power_spectrum = np.abs(fft_result)**2 / num_samples
    positive_frequencies = frequencies[:num_samples//2]
    positive_power_spectrum = power_spectrum[:num_samples//2]
    
    plt.figure()
    plt.plot(positive_frequencies, positive_power_spectrum, label=title, color=color)
    plt.title(f'Power Spectrum: {title}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

def perform_pca_and_plot(data: np.ndarray, n_components: int = 3):
    """
    Performs PCA and plots the first three principal components.
    """
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    
    # Plotting PC1 vs PC2
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    plt.title('PCA: PC1 vs PC2')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

    # Plotting PC1 vs PC3
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 2])
    plt.title('PCA: PC1 vs PC3')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 3')
    plt.grid(True)
    plt.show()

    # Plotting PC2 vs PC3
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_data[:, 1], transformed_data[:, 2])
    plt.title('PCA: PC2 vs PC3')
    plt.xlabel('Principal Component 2')
    plt.ylabel('Principal Component 3')
    plt.grid(True)
    plt.show()

    return transformed_data

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