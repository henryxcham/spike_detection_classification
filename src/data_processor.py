import os
import requests
import zipfile
import numpy as np
import spikeinterface.full as si
import scipy.signal as signal
from pathlib import Path

def download_and_extract_data(url: str, output_path: str = "data") -> Path:
    """
    Downloads and extracts a zip file from a given URL.

    Args:
        url (str): The URL of the zip file.
        output_path (str): The directory to save the data in.

    Returns:
        Path: The path to the extracted data directory.
    """
    download_dir = Path(output_path)
    download_dir.mkdir(exist_ok=True)
    zip_path = download_dir / "openephys_recording.zip"
    
    print("Downloading data...")
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        
        os.remove(zip_path)
        print("Download and extraction complete.")
        return download_dir / "openephys_raw"
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return None
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
        return None

def load_openephys_data(data_dir: Path, stream_id: str = "0"):
    """
    Loads Open Ephys data using SpikeInterface.

    Args:
        data_dir (Path): The directory containing the Open Ephys data.
        stream_id (str): The ID of the data stream to load.

    Returns:
        si.BaseRecording: The loaded recording object.
    """
    try:
        rec = si.read_openephys(data_dir.as_posix(), stream_id=stream_id)
        return rec
    except Exception as e:
        print(f"Error loading Open Ephys data: {e}")
        return None

def apply_bandpass_filter(data: np.ndarray, sample_rate: float, lowcut: float, highcut: float, order: int = 3) -> np.ndarray:
    """
    Applies a Butterworth bandpass filter to a 1D signal.
    
    This function combines the logic of the high-pass and low-pass filters
    into a single, more general function.

    Args:
        data (np.ndarray): The input 1D signal.
        sample_rate (float): The sampling frequency in Hz.
        lowcut (float): The lower cutoff frequency for the filter in Hz.
        highcut (float): The upper cutoff frequency for the filter in Hz.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered signal.
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(N=order, Wn=[low, high], btype='bandpass', analog=False)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

def apply_notch_filters(data: np.ndarray, sample_rate: float, frequencies: list, quality_factor: float = 30.0) -> np.ndarray:
    """
    Applies a series of notch filters to remove specific frequencies.
    
    This function makes the notch filtering process more efficient by
    applying all notch filters in a single loop, eliminating repetitive code.

    Args:
        data (np.ndarray): The input 1D signal.
        sample_rate (float): The sampling frequency in Hz.
        frequencies (list): A list of frequencies to remove in Hz.
        quality_factor (float): A measure of the filter's narrowness.

    Returns:
        np.ndarray: The filtered signal.
    """
    filtered_data = data.copy()
    for freq in frequencies:
        b, a = signal.iirnotch(w0=freq, Q=quality_factor, fs=sample_rate)
        filtered_data = signal.lfilter(b, a, filtered_data)
    return filtered_data