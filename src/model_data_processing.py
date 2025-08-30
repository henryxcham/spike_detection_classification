import numpy as np
import torch
from sklearn.metrics import f1_score

from src.utils import load_and_concatenate_npy, normalize_data


def create_datasets(background_paths, spikes_paths):
    background_array = load_and_concatenate_npy(background_paths)
    spikes_array = load_and_concatenate_npy(spikes_paths)
    
    if background_array.size == 0 and spikes_array.size == 0:
        return None, None
    
    X_np = np.concatenate([background_array, spikes_array], axis=0)
    y_np = np.concatenate([
        np.zeros(background_array.shape[0]), 
        np.ones(spikes_array.shape[0])
    ], axis=0)
    
    X_tensor = torch.from_numpy(X_np).float()
    y_tensor = torch.from_numpy(y_np).long()
    
    return X_tensor, y_tensor


def validate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='macro')