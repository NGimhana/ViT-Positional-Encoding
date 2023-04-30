import torch
from transformers import ViTForImageClassification, ViTConfig

class ViT16(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(ViT16, self).__init__()
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes, 
                                           use_absolute_position_embeddings=True)
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', 
                                                               config=config)
        

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits
    
        