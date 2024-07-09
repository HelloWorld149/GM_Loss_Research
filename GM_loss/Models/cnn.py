import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels, input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the flattened output from conv layers
        conv_output_size = self._calculate_conv_output_size(input_size)
        
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def _calculate_conv_output_size(self, input_size):
        # Create a dummy tensor to calculate the output size of the conv layers
        dummy_input = torch.zeros(1, 1, input_size)
        dummy_output = self.pool(self.pool(self.conv2(self.conv1(dummy_input))))
        print(f"Conv output size: {dummy_output.numel()}")
        return dummy_output.numel()

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = x.unsqueeze(1)  # Adding channel dimension
        print(f"Shape after unsqueeze: {x.shape}")
        x = self.pool(torch.relu(self.conv1(x)))
        print(f"Shape after conv1 and pool: {x.shape}")
        x = self.pool(torch.relu(self.conv2(x)))
        print(f"Shape after conv2 and pool: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"Shape after flattening: {x.shape}")
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Example usage
input_size = 30  # Example feature size, adjust as necessary
input_channels = 1
model = CNN(input_channels, input_size)

# Assuming X_train_tensor is your input tensor
X_train_tensor = torch.randn(32, input_size)  # Example batch of data, adjust as necessary
output = model(X_train_tensor)
print(f"Output shape: {output.shape}")
