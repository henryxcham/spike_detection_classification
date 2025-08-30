import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ShallowAutoencoder
from src.utils import load_and_concatenate_npy
from src.cluster import train_autoencoder

def train_ae():
    # 1. Load Data
    spikes_file_paths = [
        '../data/spikes/channel_spikes_9.npy',
        '../data/spikes/channel_spikes_16.npy',
        '../data/spikes/channel_spikes_33.npy',
        '../data/spikes/channel_spikes_11.npy',
        '../data/spikes/channel_spikes_40.npy'
    ]
    all_spikes_np = load_and_concatenate_npy(spikes_file_paths)
    all_spikes_tensor = torch.from_numpy(all_spikes_np).float()
    print(f"Total number of spikes loaded: {all_spikes_tensor.shape[0]}")

    # 2. Train Autoencoder
    autoencoder_model = ShallowAutoencoder()
    trained_autoencoder = train_autoencoder(autoencoder_model, all_spikes_tensor)
    
    # 3. Save the model
    torch.save(trained_autoencoder.state_dict(), 'best_autoencoder_model.pth')
    print("Trained autoencoder model saved as 'best_autoencoder_model.pth'.")

if __name__ == "__main__":
    train_ae()