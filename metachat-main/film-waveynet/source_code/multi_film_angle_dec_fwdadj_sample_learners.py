import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, num_features, num_conditions=3):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.num_conditions = num_conditions
        self.film_layer = nn.Linear(num_conditions, num_features * 2)

    def forward(self, x, wavelength, angle, time_norm):
        # wavelength shape: (batch_size, 1)
        # angle shape: (batch_size, 1)
        # time_norm shape: (batch_size, 1)
        # x shape: (batch_size, num_features, height, width)
        
        # Concatenate conditions (wavelength, angle, normalized time/switch state)
        conditions = torch.cat([wavelength.unsqueeze(1), angle.unsqueeze(1), time_norm.unsqueeze(1)], dim=1)
        
        # Generate FiLM parameters
        params = self.film_layer(conditions)
        gamma, beta = torch.chunk(params, 2, dim=1)
        
        # Reshape for broadcasting
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        return gamma * x + beta


class UNet(nn.Module):
    '''
    UNet architecture
    '''

    def __init__(self, net_depth, block_depth, init_num_kernels, input_channels, output_channels, dropout):

      super(UNet, self).__init__()

      self.input_channels = input_channels  # Should be 3 (1 structure + 2 source)
      self.output_channels = output_channels

      self.init_num_kernels = init_num_kernels
      self.block_depth = block_depth
      self.net_depth = net_depth

      self.conv_layers = nn.ModuleList([])
      self.bn_layers = nn.ModuleList([])
      self.dropout = nn.Dropout2d(p=dropout)
      self.film_layers = nn.ModuleList([])

      # Encoder path (no FiLM layers)
      for d in range(self.net_depth):

        curr_kernels = self.init_num_kernels * 2**d

        for b in range(self.block_depth):

          #Account for first layer taking in the low-channel input
          if(d==0 and b==0):
            self.conv_layers.append(nn.Conv2d(self.input_channels,curr_kernels, kernel_size=3, padding=1))

          else:
            if(b==0):
              prev_kernels = self.init_num_kernels * 2**(d-1)
              self.conv_layers.append(nn.Conv2d(prev_kernels,curr_kernels, kernel_size=3, padding=1))

            else:
              self.conv_layers.append(nn.Conv2d(curr_kernels,curr_kernels, kernel_size=3, padding=1))

          self.bn_layers.append(nn.BatchNorm2d(curr_kernels))

      # Decoder path (with FiLM layers)
      for d in range(self.net_depth-1):
        curr_kernels = self.init_num_kernels * 2**(self.net_depth-d-2)

        for b in range(self.block_depth):

          #Take care of the extra channels from concatenating the upsampled result from the previous block

          if(b==0):
            self.conv_layers.append(nn.Conv2d(curr_kernels*3,curr_kernels, kernel_size=3, padding=1))
          else:
            self.conv_layers.append(nn.Conv2d(curr_kernels,curr_kernels, kernel_size=3, padding=1))

          self.bn_layers.append(nn.BatchNorm2d(curr_kernels))
          self.film_layers.append(FiLM(curr_kernels))

      #One last convolution layer, for calculating the output from the processed input
      self.conv_layers.append(nn.Conv2d(self.init_num_kernels,self.output_channels, kernel_size=3, padding=1))

    #   # Add weight initialization
    #   def init_weights(m):
    #       if isinstance(m, nn.Conv2d):
    #           nn.init.kaiming_normal_(m.weight)
    #           if m.bias is not None:
    #               nn.init.zeros_(m.bias)
    #       elif isinstance(m, nn.Linear):
    #           nn.init.xavier_uniform_(m.weight)
    #           if m.bias is not None:
    #               nn.init.zeros_(m.bias)

    #   self.apply(init_weights)

    #   # Initialize FiLM layers with small weights
    #   for film in self.film_layers:
    #       nn.init.uniform_(film.film_layer.weight, -0.01, 0.01)
    #       nn.init.zeros_(film.film_layer.bias)

    def forward(self, x, wavelength, angle, time_norm):

      batch_size, _, height, width = x.shape

      shortcut_list = []  #stores shortcut layers
      conv_counter = 0
      bn_counter = 0
      film_counter = 0

      # Encoder path (no FiLM)
      for d in range(self.net_depth):

        for b in range(self.block_depth):
          features = self.conv_layers[conv_counter](x if (d == 0 and b == 0) else features)
          conv_counter+=1

          #The first convolution result of each block is added as a residual connection at the end
          if(b==0):
            res_connection = features

          features = self.bn_layers[bn_counter](features)
          bn_counter+=1

          if(b==self.block_depth-1):
            features = features + res_connection

          features = F.leaky_relu(features)
          features = self.dropout(features)

          if(b==self.block_depth-1 and d!=self.net_depth-1):
            shortcut_list.append(features)

        if(d != self.net_depth-1):
          features = F.max_pool2d(input=features, kernel_size=2)

      # Decoder path (with FiLM)
      for d in range(self.net_depth-1):

        #Convention: (N,C,H,W)
        features = torch.cat(( F.interpolate(features, scale_factor=(2,2), mode='nearest') , shortcut_list.pop() ),dim=1)

        for b in range(self.block_depth):

          features = self.conv_layers[conv_counter](features)
          conv_counter+=1

          if(b==0):
            res_connection = features

          features = self.bn_layers[bn_counter](features)
          features = self.film_layers[film_counter](features, wavelength, angle, time_norm)
          bn_counter+=1
          film_counter+=1

          if(b==self.block_depth-1):
            features = features + res_connection

          features = F.leaky_relu(features)
          features = self.dropout(features)

      output = self.conv_layers[conv_counter](features)

      return output


def upgrade_film_state_dict(state_dict, num_conditions_new=3, num_conditions_old=2):
    """
    Upgrade a legacy state dict from 2-condition FiLM to 3-condition FiLM
    by zero-padding the weight matrix of each FiLM linear layer.

    The pretrained weights W have shape (2*num_features, 2). We expand to
    (2*num_features, 3) by appending a zero column for the new time condition.
    The bias is unchanged. This guarantees:
        W_new @ [wl, angle, t]^T  ==  W_old @ [wl, angle]^T + 0*t
    so the model output is mathematically identical regardless of time_norm.

    Parameters
    ----------
    state_dict : dict
        Model state dict (from a 2-condition pretrained checkpoint).
    num_conditions_new : int
        Target number of conditions (default 3: wavelength, angle, time).
    num_conditions_old : int
        Number of conditions in the pretrained model (default 2).

    Returns
    -------
    state_dict : dict
        Updated state dict with zero-padded FiLM weights.
    upgraded_keys : list[str]
        List of parameter keys that were upgraded.
    """
    upgraded_keys = []
    for key in list(state_dict.keys()):
        if 'film_layer.weight' in key:
            old_weight = state_dict[key]  # shape: (out_features, num_conditions_old)
            if old_weight.shape[1] == num_conditions_old:
                # Zero-pad: append (num_conditions_new - num_conditions_old) zero columns
                pad_cols = num_conditions_new - num_conditions_old
                zero_pad = torch.zeros(old_weight.shape[0], pad_cols,
                                       dtype=old_weight.dtype, device=old_weight.device)
                new_weight = torch.cat([old_weight, zero_pad], dim=1)
                state_dict[key] = new_weight
                upgraded_keys.append(key)
    return state_dict, upgraded_keys


def load_legacy_checkpoint(checkpoint_path, net_depth, block_depth, init_num_kernels,
                           input_channels=3, output_channels=2, dropout=0,
                           map_location=None):
    """
    Load a pretrained 2-condition FiLM WaveY-Net checkpoint into the new
    3-condition architecture without retraining.

    Handles both checkpoint formats:
      - Full model object: checkpoint['model'] is a nn.Module
      - State dict only: checkpoint['state_dict'] is an OrderedDict

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt checkpoint file.
    net_depth, block_depth, init_num_kernels : int
        Architecture hyperparameters (must match the pretrained model).
    input_channels : int
        Number of input channels (default 3: 1 structure + 2 source).
    output_channels : int
        Number of output channels (default 2: Hz_real, Hz_imag).
    dropout : float
        Dropout rate (default 0).
    map_location : str or torch.device, optional
        Device mapping for torch.load.

    Returns
    -------
    model : UNet
        The new 3-condition UNet with pretrained weights loaded.
    checkpoint : dict
        The full checkpoint dict (for extracting epoch, optimizer, etc.).
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Determine if checkpoint contains a full model or just state_dict
    if 'model' in checkpoint and isinstance(checkpoint['model'], nn.Module):
        old_model = checkpoint['model']
        old_state_dict = old_model.state_dict()
    elif 'state_dict' in checkpoint:
        old_state_dict = checkpoint['state_dict']
    else:
        raise ValueError("Checkpoint does not contain 'model' or 'state_dict' key.")

    # Check if upgrade is needed by inspecting the first FiLM layer weight
    needs_upgrade = False
    for key in old_state_dict:
        if 'film_layer.weight' in key:
            if old_state_dict[key].shape[1] == 2:
                needs_upgrade = True
            break

    # Create new 3-condition model
    model = UNet(net_depth, block_depth, init_num_kernels,
                 input_channels=input_channels,
                 output_channels=output_channels,
                 dropout=dropout).float()

    if needs_upgrade:
        upgraded_state_dict, upgraded_keys = upgrade_film_state_dict(old_state_dict)
        model.load_state_dict(upgraded_state_dict)
        print(f"[Legacy Checkpoint] Upgraded {len(upgraded_keys)} FiLM layers from "
              f"2-condition to 3-condition (zero-padded time column).")
        print(f"  Upgraded keys: {upgraded_keys}")
    else:
        model.load_state_dict(old_state_dict)
        print("[Checkpoint] Loaded 3-condition checkpoint directly (no upgrade needed).")

    # Replace the model in checkpoint so downstream code works seamlessly
    checkpoint['model'] = model

    return model, checkpoint