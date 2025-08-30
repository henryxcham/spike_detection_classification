# Spike Detection Classification Project

This repository contains code for a neuroscience project that applys deep learning techniques to spike sorting from Open Ephys data. The project is divided into three main stages:

1.  **Data Preprocessing and Spike Extraction**: Loading raw electrophysiology data, applying filters to remove noise, and detecting neuronal spikes based on a voltage threshold.
2.  **Spike Classification**: Using machine learning models (CNN and GRU) to distinguish true neuronal spikes from background noise.
3.  **Spike Clustering**: Using dimensionality reduction (PCA, Autoencoder) and clustering algorithms (DBSCAN) to classify neuronal spikes into individual neurons.

## 🚀 Repository Structure

* `data/`: Raw data downloaded and extracted from the provided URL.
* `src/`: Contains core Python modules for data processing, spike detection, and utility functions.
* `notebooks/`: Jupyter notebooks that walk through the analysis workflow step-by-step.
* `snippets/`: The output directory for classified spike and background snippets.

## 📦 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install the required packages:**
    (You'll need to create a `requirements.txt` file based on the packages you used: `numpy`, `matplotlib`, `scipy`, `requests`, `spikeinterface`, `probeinterface`, `scikit-learn`)
    ```bash
    pip install -r requirements.txt
    ```

## 📈 Usage

1.  **Run the Jupyter Notebooks:**
    Navigate to the `notebooks` directory and launch Jupyter Lab or Notebook.
    ```bash
    jupyter notebook notebooks/01_data_loading_and_spike_extraction.ipynb
    ```
    The notebook will guide you through the process, from data download and filtering to spike extraction and saving the results. The code is structured to call functions from the `src/` directory, making the notebook clean and easy to follow.

## 🧠 Core Functions

* `src/data_processor.py`: Handles data downloading, loading, and filtering. Key functions include `download_and_extract_data`, `apply_bandpass_filter`, and `apply_notch_filters`.
* `src/spike_detector.py`: Contains the logic for spike detection. The `dynamic_spike_capture_from_signal` function implements an adaptive thresholding method for more robust spike detection.
* `src/utils.py`: Provides helper functions for plotting power spectra and performing PCA on the data.
