import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def evaluate_model(model: nn.Module, model_path: str, data_loader: DataLoader, data_name: str):
    """
    Loads a trained model and evaluates it on a dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Cannot evaluate.")
        return

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    final_accuracy = accuracy_score(all_labels, all_preds)
    final_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    final_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    final_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\n--- Best Model {data_name} Metrics ---")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Macro Precision: {final_precision:.4f}")
    print(f"Macro Recall: {final_recall:.4f}")
    print(f"Macro F1 Score: {final_f1:.4f}")