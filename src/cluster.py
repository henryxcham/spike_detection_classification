import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import os

def train_autoencoder(model, data, epochs=200, lr=0.001):
    """
    Trains the autoencoder model on the provided data.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)
    
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting autoencoder training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        reconstructed = model(data)
        loss = loss_function(reconstructed, data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("Autoencoder training finished!")
    return model

def plot_reconstruction(model, original_data):
    """
    Plots a random sample's original waveform and its autoencoder reconstruction.
    """
    with torch.no_grad():
        random_index = np.random.randint(0, len(original_data))
        sample = original_data[random_index:random_index+1].numpy().flatten()
        
        device = next(model.parameters()).device
        sample_tensor = torch.from_numpy(sample).float().unsqueeze(0).to(device)
        
        encoded_sample = model.encoder(sample_tensor)
        decoded_sample = model.decoder(encoded_sample).cpu().numpy().flatten()
        
        plt.figure(figsize=(12, 6))
        plt.plot(sample, label='Original Data', color='blue')
        plt.plot(decoded_sample, label='Reconstructed Data', color='red', linestyle='--')
        plt.title('Autoencoder Reconstruction')
        plt.xlabel('Data Point Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

def apply_and_plot_dbscan(data_embeddings, title, x_label, y_label, z_label, eps, min_samples):
    """
    Applies DBSCAN to embeddings and plots the results in a 3D scatter plot.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data_embeddings)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels)
    
    for k in unique_labels:
        cluster_mask = (labels == k)
        
        if k == -1:
            ax.scatter(data_embeddings[cluster_mask, 0], 
                       data_embeddings[cluster_mask, 1], 
                       data_embeddings[cluster_mask, 2],
                       s=5, alpha=0.6, color='k', label='Noise')
        else:
            ax.scatter(data_embeddings[cluster_mask, 0], 
                       data_embeddings[cluster_mask, 1], 
                       data_embeddings[cluster_mask, 2],
                       s=5, alpha=0.6, label=f'Cluster {k}')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.legend()
    ax.view_init(elev=20, azim=90)
    plt.show()