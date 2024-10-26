import torch 
import torch.nn as nn 


# Why use Swish activation function?
# 1. Swish is a smooth function, which is differentiable everywhere.
# 2. Swish is non-monotonic, which can help the model to learn complex patterns.
# 3. Swish is easy to implement, and it can be used in any neural network.
# 4. Swish is computationally efficient, which can be implemented using a single line of code.
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# We use cnn model as the base model for the EBM model
# The CNN model output the nagative energy of the input image
class CNNModel(nn.Module):
    def __init__(self, hidden_features = 32, out_dim = 1, **kwargs):
        super().__init__()
        
        c_hid1 = hidden_features // 2 
        c_hid2 = hidden_features 
        c_hid3 = hidden_features * 2 
    
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4), # [16x16] - Larger padding to get 32x32 image
                Swish(),
                nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1), #  [8x8]
                Swish(),
                nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1), # [4x4]
                Swish(),
                nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1), # [2x2]
                Swish(),
        )
        
        cnn_output_dim = self._cnn_output_dim((1, 28, 28))
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, c_hid3),
            Swish(),
            nn.Linear(c_hid3, out_dim)
        )
        
    
    def _cnn_output_dim(self, input_shape):
        return self.cnn_layers(torch.zeros(1, *input_shape)).view(1, -1).shape[1]
    
    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim = -1)
        return x 