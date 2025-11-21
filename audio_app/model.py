# In model.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as TV

class ViTModel(nn.Module):
    """
    This class defines the Vision Transformer architecture, adapted for 
    1-channel spectrogram inputs.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Load the pre-trained Vision Transformer (ViT) architecture.
        # We set weights=None because we will be loading our own fine-tuned weights.
        self.vit = models.vit_b_16(weights=None)
        
        # --- Modifications for Spectrograms ---
        
        # 1. Modify the first convolutional layer (patch embedding) to accept
        #    1-channel (grayscale) spectrograms instead of 3-channel RGB images.
        #    Original layer: nn.Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
        self.vit.conv_proj = nn.Conv2d(
            in_channels=1, 
            out_channels=768, 
            kernel_size=(16, 16), 
            stride=(16, 16)
        )
        
        # 2. Add a resizing transform because the pre-trained ViT expects
        #    a fixed input size of 224x224 pixels.
        self.resizer = TV.Resize((224, 224), antialias=True)
        
        # 3. Replace the final classifier head to output the correct number
        #    of classes for our problem (2: fake or real).
        num_final_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_final_features, num_classes)

    def forward(self, x):
        """Defines the forward pass of the model."""
        # The input 'x' is a spectrogram tensor.
        # First, resize it to the 224x224 size that ViT expects.
        x = self.resizer(x)
        # Then, pass it through the ViT model.
        return self.vit(x)