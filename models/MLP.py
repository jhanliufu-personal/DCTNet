import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple 3-layer Multi-Layer Perceptron for phase estimation baseline.
    This model serves as a simple baseline to compare against the DCTNN.
    """
    
    def __init__(self, input_size=512, hidden_size=256, output_size=1, dropout_rate=0.2):
        """
        Initialize MLP model.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden layers
            output_size (int): Size of output (1 for real or imaginary part)
            dropout_rate (float): Dropout rate for regularization
        """
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        return self.model(x)


class MLPComplex(nn.Module):
    """
    MLP model that outputs both real and imaginary parts simultaneously.
    This can be more efficient than training two separate models.
    """
    
    def __init__(self, input_size=512, hidden_size=256, dropout_rate=0.2):
        """
        Initialize complex MLP model.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden layers
            dropout_rate (float): Dropout rate for regularization
        """
        super(MLPComplex, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Separate heads for real and imaginary parts
        self.real_head = nn.Linear(hidden_size, 1)
        self.imag_head = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Forward pass through the complex MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2) where [:, 0] is real and [:, 1] is imag
        """
        shared_features = self.shared_layers(x)
        
        real_part = self.real_head(shared_features)
        imag_part = self.imag_head(shared_features)
        
        return torch.cat([real_part, imag_part], dim=1)
    
    def forward_real(self, x):
        """Forward pass for real part only."""
        shared_features = self.shared_layers(x)
        return self.real_head(shared_features)
    
    def forward_imag(self, x):
        """Forward pass for imaginary part only.""" 
        shared_features = self.shared_layers(x)
        return self.imag_head(shared_features)