import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=0.5)

        dummy_input = torch.zeros(1, 1, input_size)
        dummy_output = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(dummy_input))))))
        flattened_size = dummy_output.view(1, -1).size(1)

        self.layer_norm1 = nn.LayerNorm(flattened_size)
        self.fc1 = nn.Linear(in_features=flattened_size, out_features=32)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.layer_norm1(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

class GRUClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GRUClassifier, self).__init__()
        
        self.gru1 = nn.GRU(input_size=1, hidden_size=16, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.gru2 = nn.GRU(input_size=16, hidden_size=32, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(in_features=32, out_features=32)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        gru1_out, _ = self.gru1(x)
        x = self.dropout1(gru1_out)
        
        gru2_out, h_n = self.gru2(x)
        x = self.dropout2(gru2_out)

        x = h_n[-1]
        
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

class ShallowAutoencoder(nn.Module):
    def __init__(self, input_size=42, embedding_size=3):
        super(ShallowAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 21),
            nn.ReLU(),
            nn.Linear(21, embedding_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 21),
            nn.ReLU(),
            nn.Linear(21, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded