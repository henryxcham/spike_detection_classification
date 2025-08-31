import numpy as np
import os
from collections import deque
from pathlib import Path

def capture_spikes_from_signal(
    signal: np.ndarray, 
    sample_rate: float, 
    threshold_V: float,
    snippet_before_ms: float = 0.4, 
    snippet_after_ms: float = 1.0, 
    refractory_period_ms: float = 1.0
):
    """
    Captures neuronal spikes from a signal when it crosses a negative threshold.

    Args:
        signal (np.ndarray): The input 1D signal array, in Volts.
        sample_rate (float): The sampling frequency of the signal in Hz.
        threshold_V (float): The negative voltage threshold.
        snippet_before_ms (float): Duration of the snippet to capture before.
        snippet_after_ms (float): Duration of the snippet to capture after.
        refractory_period_ms (float): Refractory period in milliseconds.

    Yields:
        tuple: (signal snippet, trigger time in seconds).
    """
    samples_before = int((snippet_before_ms / 1000.0) * sample_rate)
    samples_after = int((snippet_after_ms / 1000.0) * sample_rate)
    refractory_samples = int((refractory_period_ms / 1000.0) * sample_rate)
    num_samples = len(signal)
    
    trigger_indices = np.where((signal[1:] < threshold_V) & (signal[:-1] >= threshold_V))[0] + 1
    
    i = 0
    while i < len(trigger_indices):
        trigger_idx = trigger_indices[i]
        start_idx = trigger_idx - samples_before
        end_idx = trigger_idx + samples_after
        
        if 0 <= start_idx < end_idx < num_samples:
            spike_snippet = signal[start_idx:end_idx]
            trigger_time_s = trigger_idx / sample_rate
            yield spike_snippet, trigger_time_s
            
        next_valid_idx = trigger_idx + refractory_samples
        j = i + 1
        while j < len(trigger_indices) and trigger_indices[j] < next_valid_idx:
            j += 1
        i = j

def dynamic_spike_capture_from_signal(
    signal: np.ndarray, 
    sample_rate: float, 
    snippet_before_ms: float = 0.4, 
    snippet_after_ms: float = 1.0, 
    refractory_period_ms: float = 1.0, 
    start_threshold_std: int = 5
):
    """
    Captures neuronal spikes using a dynamically adjusting threshold.

    Args:
        signal (np.ndarray): The input 1D signal array.
        sample_rate (float): The sampling frequency in Hz.
        snippet_before_ms (float): Duration of the snippet to capture before.
        snippet_after_ms (float): Duration of the snippet to capture after.
        refractory_period_ms (float): Refractory period in milliseconds.
        start_threshold_std (int): Initial threshold in standard deviations.

    Yields:
        tuple: (signal snippet, trigger time in seconds).
    """
    samples_before = int((snippet_before_ms / 1000.0) * sample_rate)
    samples_after = int((snippet_after_ms / 1000.0) * sample_rate)
    refractory_samples = int((refractory_period_ms / 1000.0) * sample_rate)
    
    threshold_V = 0.0
    spike_counts_per_sec = deque(maxlen=5)
    last_trigger_idx = -refractory_samples
    chunk_size_samples = int(1.0 * sample_rate)
    
    for i in range(0, len(signal), chunk_size_samples):
        chunk_start_idx = i
        chunk_end_idx = min(i + chunk_size_samples, len(signal))
        current_chunk = signal[chunk_start_idx:chunk_end_idx]
        
        if i == 0:
            std_V = np.std(current_chunk)
            threshold_V = np.mean(current_chunk) - start_threshold_std * std_V
        else:
            std_V = np.std(current_chunk)

        chunk_trigger_indices = np.where((current_chunk[1:] < threshold_V) & (current_chunk[:-1] >= threshold_V))[0] + 1
        
        current_chunk_spikes = 0
        for trigger_idx_in_chunk in chunk_trigger_indices:
            full_signal_idx = chunk_start_idx + trigger_idx_in_chunk
            if full_signal_idx >= last_trigger_idx + refractory_samples:
                start_idx = full_signal_idx - samples_before
                end_idx = full_signal_idx + samples_after
                
                if 0 <= start_idx < end_idx < len(signal):
                    spike_snippet = signal[start_idx:end_idx]
                    trigger_time_s = full_signal_idx / sample_rate
                    yield spike_snippet, trigger_time_s
                    current_chunk_spikes += 1
                    last_trigger_idx = full_signal_idx
        
        spike_counts_per_sec.append(current_chunk_spikes)
        if len(spike_counts_per_sec) == 5:
            total_spikes_last_5s = sum(spike_counts_per_sec)
            if total_spikes_last_5s < 5:
                threshold_V *= 1.1 # Adjust threshold to be less negative
            elif total_spikes_last_5s > 500:
                threshold_V *= 0.9 # Adjust threshold to be more negative

def save_snippets(snippets: np.ndarray, output_dir: str, filename: str):
    """
    Saves a numpy array of snippets to a file.

    Args:
        snippets (np.ndarray): The array of snippets.
        output_dir (str): The output directory.
        filename (str): The name of the file to save.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    np.save(file_path, snippets)
    print(f"Saved {len(snippets)} snippets to {file_path}")