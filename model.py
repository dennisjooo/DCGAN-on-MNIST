# Importing the libraries
import torch.nn as nn

# Custom reshape layer
class Reshape(nn.Module):
    def __init__(self, *args):
        
        # Inherits all that good stuff from nn.Module
        super(Reshape, self).__init__()

        # Shape is a tuple of integers
        self.shape = args

    def forward(self, x):
        # Returns a view of the input tensor
        return x.view(self.shape)
    

# Generator
class Generator(nn.Module):
    def __init__(self, coding_size=100):

        # Inherits all that good stuff from nn.Module
        super(Generator, self).__init__()

        # Defining the layers
        self.layer_stack = nn.Sequential(
            nn.Linear(coding_size, 128 * 7 * 7), # (N, 100) -> (N, 128 * 7 * 7)
            Reshape(-1, 128, 7, 7), # (N, 128 * 7 * 7) -> (N, 128, 7, 7)
            nn.BatchNorm2d(128), # 2D batch normalization
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (N, 128, 7, 7) -> (N, 64, 14, 14)
            nn.ReLU(), # ReLU activation
            nn.BatchNorm2d(64), # 2D batch normalization
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1), # (N, 64, 14, 14) -> (N, 1, 28, 28)
            nn.Tanh() # Tanh activation
        )

    def forward(self, x):

        # Returns the output of the layer stack
        return self.layer_stack(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):

        # Inherits all that good stuff from nn.Module
        super(Discriminator, self).__init__()
        
        # Defining the layers
        self.layer_stack = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), # (N, 1, 28, 28) -> (N, 64, 14, 14)
            nn.LeakyReLU(0.2), # Leaky ReLU activation
            nn.Dropout2d(0.3), # 2D dropout
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (N, 64, 14, 14) -> (N, 128, 7, 7)
            nn.LeakyReLU(0.2), # Leaky ReLU activation
            nn.Dropout2d(0.3), # 2D dropout
            Reshape(-1, 128 * 7 * 7), # (N, 128, 7, 7) -> (N, 128 * 7 * 7)
            nn.Linear(128 * 7 * 7, 1), # (N, 128 * 7 * 7) -> (N, 1)
            nn.Sigmoid() # Sigmoid activation
        )

    def forward(self, x):

        # Returns the output of the layer stack
        return self.layer_stack(x)