import torch
import torch.nn as nn
import torch.nn.functional as F

def triangle_wave(x, period, phase=0):
    return 1 - 2 * torch.abs(torch.fmod(x / period + phase, 1) - 0.5)

class ParallelEncoder(nn.Module):
    def __init__(self, device, encoding_config, num_groups, input_dims=2):
        super().__init__()
        dim = 32
        self.dim = dim
        self.num_groups = num_groups
        inv_freq = 100 / (100 ** (torch.arange(0, dim, 2).float() / dim)).to(device)
        self.register_buffer("inv_freq", inv_freq)
        self.output_dim = dim

    def forward(self, y):
        # y shape: [num_models, batch_size, input_dims]
        y_scaled = y * self.inv_freq.unsqueeze(0).unsqueeze(0)
        sin = torch.sin(y_scaled)
        cos = torch.cos(y_scaled)
        return torch.cat((sin, cos), dim=-1).float()

class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Create separate weights for each group
        self.weight = nn.Parameter(
            torch.randn(num_groups, out_features, in_features, dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=2.236)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # x shape: [num_groups, batch_size, in_features]
        # Perform grouped matrix multiplication
        output = torch.bmm(x, self.weight.transpose(1, 2))
        if self.bias is not None:
            output += self.bias.unsqueeze(1)
        return output

class Network(nn.Module):
    def __init__(self, args, num_groups):
        super().__init__()
        self.args = args
        self.num_groups = num_groups
        
        self.encoding = ParallelEncoder(
            self.args.device,
            encoding_config=args.encoding,
            num_groups=num_groups
        ).to(args.device).float()
        
        # Create grouped network layers
        self.layer1 = GroupedLinear(self.encoding.output_dim, 32, num_groups, bias=False)
        self.layer2 = GroupedLinear(32, 32, num_groups, bias=False)
        self.layer3 = GroupedLinear(32, 32, num_groups, bias=False)
        self.layer4 = GroupedLinear(32, 1, num_groups, bias=False)
        
    def forward(self, x):
        # x shape: [num_groups, batch_size, input_dims]
        x = self.encoding(x)
        x = F.leaky_relu(self.layer1(x), 0.1)
        x = F.leaky_relu(self.layer2(x), 0.1)
        x = F.leaky_relu(self.layer3(x), 0.1)
        x = self.layer4(x)
        return x