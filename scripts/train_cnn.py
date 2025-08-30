import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import sys

# Add the parent directory to the path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import CNNClassifier
from src.utils import normalize_data
from src.model_data_processing import create_datasets, validate_model

def train_cnn():
    # 1. Load and Prepare Data
    background_file_paths_train = [
        '../data/spikes/channel_background_9.npy',
        '../data/spikes//channel_background_16.npy',
        '../data/spikes/channel_background_33.npy',
    ]
    spikes_file_paths_train = [
        '../data/spikes/channel_spikes_9.npy',
        '../data/spikes/channel_spikes_16.npy',
        '../data/spikes/channel_spikes_33.npy',
    ]
    background_file_paths_valid = [
        '../data/spikes/channel_background_11.npy'
    ]
    spikes_file_paths_valid = [
        '../data/spikes/channel_spikes_11.npy'
    ]

    X_train, y_train = create_datasets(background_file_paths_train, spikes_file_paths_train)
    X_valid, y_valid = create_datasets(background_file_paths_valid, spikes_file_paths_valid)

    train_mean = X_train.mean()
    train_std = X_train.std()
    X_train_normalized = normalize_data(X_train, train_mean, train_std).unsqueeze(1)
    X_valid_normalized = normalize_data(X_valid, train_mean, train_std).unsqueeze(1)

    train_dataset = TensorDataset(X_train_normalized, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    valid_dataset = TensorDataset(X_valid_normalized, y_valid)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)

    # 2. Instantiate and Train the Model
    num_classes = 2
    input_size = X_train.shape[1]
    model = CNNClassifier(input_size=input_size, num_classes=num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_f1 = 0.0
    patience = 5
    patience_counter = 0
    num_epochs = 20
    model_name = 'cnn'
    model_dir = '../model'
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = f'{model_dir}/best_{model_name}_model.pth'



    print(f"Starting {model_name} training loop...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        valid_f1 = validate_model(model, valid_loader, device)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}, Valid F1: {valid_f1:.4f}')

        if valid_f1 > best_f1:
            best_f1 = valid_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved with a new best F1 score of {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"Validation F1 did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

if __name__ == "__main__":
    train_cnn()