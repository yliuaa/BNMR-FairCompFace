import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FairWeighNet(nn.Module):
    def __init__(self, architecture, input_size, dense_sizes=[256, 512, 256], num_heads=8):
        super(FairWeighNet, self).__init__()
        modules = []

        # Add attention layer
        if architecture == "attention":
            raise NotImplementedError("Attention wrapper not implemented yet")
            self.self_attn = nn.MultiheadAttention(embed_dim=1, num_heads=num_heads)
            modules.append(self.AttentionWrapper(self.self_attn))

        # Constructing the dense layers using the provided sizes
        prev_size = input_size
        for size in dense_sizes:
            modules.extend([
                nn.Linear(prev_size, size),
                nn.ReLU()
            ])
            prev_size = size
        modules.append(nn.Linear(prev_size, 1))
        modules.append(nn.Sigmoid())
        self.net = nn.Sequential(*modules)


    # Set sequence length to be 0
    # FIXME: attention size wrapper
    class AttentionWrapper(nn.Module):
        def __init__(self, attention_layer):
            super().__init__()
            self.attention = attention_layer

        def forward(self, x):
            x = x.unsqueeze(2)
            print(x.shape)
            attn_output, _ = self.attention(x, x, x)
            return attn_output.squeeze(2)
        
    def forward(self, x):
        out = self.net(x)
        return out.squeeze(1)



model = FairWeighNet("dense", input_size=64, dense_sizes=[300, 400])
input_data = torch.randn(10, 64)
output = model(input_data)
print(output.shape)
print(output)
         
