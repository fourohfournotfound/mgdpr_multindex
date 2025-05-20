import torch
import torch.nn as nn
import torch.nn.functional as F # Added for potential use, though not in current snippet

class MultiReDiffusion(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_relation):
        super(MultiReDiffusion, self).__init__()
        self.output = output_dim
        self.fc_layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_relation)])
        self.update_layer = torch.nn.Conv2d(num_relation, num_relation, kernel_size=1)
        self.activation1 = torch.nn.PReLU()
        self.activation0 = torch.nn.PReLU()
        self.num_relation = num_relation

    def forward(self, theta_param, t_param, a_input_batched, x_input_batched):
        # theta_param: (Num_Relations, Expansion_Steps) - Parameters, not batched
        # t_param: (Num_Relations, Expansion_Steps, Num_Nodes, Num_Nodes) - Parameters, not batched
        # a_input_batched: (Batch_Size, Num_Relations, Num_Nodes, Num_Nodes) - Batched input adjacencies
        # x_input_batched: (Batch_Size, Num_Relations, Num_Nodes, Features_Input_Dim) - Batched input features

        device = x_input_batched.device
        batch_size = x_input_batched.shape[0]
        num_relations = theta_param.shape[0] # self.num_relation
        num_nodes = x_input_batched.shape[2]
        # self.output is the output feature dimension of self.fc_layers

        # Initialize diffusions tensor to store results for each batch item, relation, and node
        # Shape: (Batch_Size, Num_Relations, Num_Nodes, Features_Output_FC)
        diffusions_batch = torch.zeros(batch_size, num_relations, num_nodes, self.output, device=device)

        for b_idx in range(batch_size):
            # Get data for the current batch item
            # These are now unbatched for the inner loops:
            # a_sample: (Num_Relations, Num_Nodes, Num_Nodes)
            # x_sample: (Num_Relations, Num_Nodes, Features_Input_Dim)
            a_sample = a_input_batched[b_idx]
            x_sample = x_input_batched[b_idx]

            for rel_idx in range(num_relations):
                # current_a_rel_slice: (Num_Nodes, Num_Nodes)
                current_a_rel_slice = a_sample[rel_idx]
                # current_x_rel_slice: (Num_Nodes, Features_Input_Dim)
                current_x_rel_slice = x_sample[rel_idx]
                
                diffusion_mat = torch.zeros_like(current_a_rel_slice, device=device)
                for step_idx in range(theta_param.shape[1]): # expansion_steps
                    # t_param[rel_idx, step_idx] is (Num_Nodes, Num_Nodes)
                    diffusion_mat += theta_param[rel_idx, step_idx] * t_param[rel_idx, step_idx] * current_a_rel_slice
                
                # diffusion_feat: (Num_Nodes, Features_Input_Dim)
                diffusion_feat = torch.matmul(diffusion_mat, current_x_rel_slice)
                
                # fc_output: (Num_Nodes, Features_Output_FC)
                fc_output = self.fc_layers[rel_idx](diffusion_feat)
                
                # Store in the correct slice of the batched diffusions tensor
                diffusions_batch[b_idx, rel_idx] = self.activation0(fc_output)

        # diffusions_batch is now (Batch_Size, Num_Relations, Num_Nodes, Features_Output_FC)
        # This shape is suitable for self.update_layer (Conv2d)
        # Conv2d expects (Batch, Channels_in, H, W)
        # Here, Batch_Size, Num_Relations (as Channels_in), Num_Nodes (as H), Features_Output_FC (as W)
        
        # The original code used diffusions.unsqueeze(0) because it assumed diffusions was (R,N,F_out)
        # Now diffusions_batch is already (B,R,N,F_out), so no unsqueeze is needed.
        # Conv2d expects (batch_size, in_channels, height, width)
        # Here, in_channels = num_relation, height = num_nodes, width = output_dim (if output_dim is treated as width)
        # This might need adjustment based on how num_nodes and output_dim are intended as spatial dimensions.
        # Assuming diffusions is (num_relation, num_nodes, features_out)
        # To make it (batch_size=1, num_relation_channels, num_nodes_H, features_out_W)
        # This part seems to assume diffusions is (num_relations, num_nodes, output_dim)
        # and update_layer treats num_relations as channels.
        # If diffusions is (num_relations, num_nodes, output_dim_per_relation)
        # Conv2d expects (N, C_in, H, W). Here, N=1, C_in=num_relation.
        # The input to Conv2d should be (1, num_relation, num_nodes, self.output)
        # So, diffusions needs to be permuted if num_nodes is H and self.output is W.
        # Or, if self.output is not a spatial dimension, Conv1d might be more appropriate.
        # Given Conv2d(num_relation, num_relation, kernel_size=1), it implies operating on each (node, feature_dim) pair independently across relations.
        # Let's assume diffusions is (num_relation, num_nodes, self.output)
        # Permute to (num_nodes, self.output, num_relation) then view as (1, num_relation, num_nodes, self.output) for conv2d
        # Or more directly: diffusions.unsqueeze(0) if diffusions is (C_in, H, W)
        # If diffusions is (num_relations, num_nodes, output_dim), then diffusions.unsqueeze(0) is (1, num_relations, num_nodes, output_dim)
        # This matches the expectation of Conv2d if num_nodes is H and output_dim is W.
        
        # The paper mentions Conv2d_1x1(Delta(S_lr H_l-1 W_l^r)). Delta stacks relational node representations.
        # So, diffusions is likely (num_relation, num_nodes, output_dim_of_fc_layers)
        # update_layer is Conv2d(num_relation, num_relation, kernel_size=1)
        # Input to conv2d should be (Batch, Channels_in, H, W)
        # Here, Batch=1 (implicitly), Channels_in = num_relation. H=num_nodes, W=output_dim
        # So, diffusions.unsqueeze(0) is (1, num_relation, num_nodes, self.output) - this seems correct.
        latent_feat_batch = self.activation1(self.update_layer(diffusions_batch))
        # latent_feat_batch shape: (Batch_Size, Num_Relations_out, Num_Nodes, Features_Output_FC)
        # Since self.update_layer is Conv2d(num_relation, num_relation, ...), Num_Relations_out = Num_Relations.
        # So, latent_feat_batch is (Batch_Size, Num_Relations, Num_Nodes, Features_Output_FC)

        # The MGDPR layer expects two outputs from diffusion_layers:
        # h_diffused (which becomes the new h for next diffusion) and u_intermediate (input to retention)
        # Both should have the same shape: (Batch_Size, Num_Relations, Num_Nodes, Features_Output_FC)
        # In the original paper/notebook, h_diffused and u_intermediate are often the same tensor or derived closely.
        # Here, latent_feat_batch is the result after the Conv2d update, and diffusions_batch is before it.
        # Let's return latent_feat_batch as the primary "diffused" representation,
        # and diffusions_batch as the intermediate one if needed (though often they are the same or u is latent_feat).
        # The MGDPR model uses the second output (u_intermediate) as input to ParallelRetention.
        # The first output (h_diffused) becomes the new `h` for the next MGDPR layer's diffusion part.
        # It's common for u_intermediate to be the same as h_diffused.
        # Let's assume latent_feat_batch is the refined representation to be used for both purposes.
        return latent_feat_batch, latent_feat_batch # Return (h_diffused, u_intermediate)


class ParallelRetention(torch.nn.Module):
    def __init__(self, time_dim, in_dim, inter_dim, out_dim): # time_dim seems to be num_nodes from usage
        super(ParallelRetention, self).__init__()
        # The paper's Parallel Retention (Eq 4) takes Z (node features over time/relations).
        # Z has dimensions like (num_nodes, feature_dim_from_diffusion) or (time_steps, feature_dim)
        # In the MGDPR forward pass, `u` is passed to retention. `u` is (num_relation, num_nodes, diffusion_output_dim)
        # The retention_layers are initialized with retention[3*i], retention[3*i+1], retention[3*i+2]
        # These correspond to in_dim, inter_dim, out_dim for retention.
        # The `time_dim` parameter in init is `time_dim` from MGDPR init, which is `time_steps` (window size).
        # However, in forward(self, x, d_gamma): x is `u` which is (num_relation, num_nodes, features)
        # x = x.view(self.time_dim, -1) -> this reshapes `u`
        # This implies `time_dim` in ParallelRetention's init should match the first dimension of the reshaped `u`.
        # If `u` is (num_relation, num_nodes, features), and reshaped to (self.time_dim, something),
        # then num_relation must be self.time_dim. This seems to be a mismatch with `time_steps`.
        # Let's look at the paper's Fig 1 and Eq 4. eta(H_l) where H_l is latent diffusion representation.
        # H_l is (num_relation, num_nodes, features).
        # Parallel retention in paper: Q = ZW_Q, K = ZW_K, V = ZW_V. D is decay matrix (T x T).
        # eta(Z) = phi((QK^T . D)V). Z is (T x feature_dim) if T is sequence length.
        # In the code, x (which is `u`) is (num_relation, num_nodes, features_from_diffusion).
        # x.view(self.time_dim, -1) means num_relation is treated as the "time" or sequence dimension for retention.
        # So, self.time_dim in ParallelRetention should correspond to num_relation.
        # And in_dim should be num_nodes * features_from_diffusion. This seems complex.

        # Let's re-evaluate based on `demo.ipynb` usage:
        # `retention_layers` are init with `ParallelRetention(time_dim, ...)` where `time_dim` is `time_steps` (window size).
        # `u` passed to `retention_layers[l](u, self.D_gamma)` is `(num_relation, num_nodes, features)`.
        # Inside `ParallelRetention.forward(self, x, d_gamma)`:
        # `x` is `u`. `x.shape[1]` is `num_nodes`.
        # `x = x.view(self.time_dim, -1)`: this means `x` (num_relation, num_nodes, features) is reshaped.
        # For this view to work, `num_relation * num_nodes * features == self.time_dim * something`.
        # This implies `self.time_dim` (which is `time_steps` from MGDPR init) is the first dimension of the view.
        # This means `num_relation` is being treated as the sequence length for retention.
        # And `in_dim` for Q,K,V layers is `(num_nodes * features_from_diffusion)`.
        # `d_gamma` is (time_dim, time_dim) i.e. (num_relation, num_relation).
        # This interpretation means retention operates over the relations.

        self.time_dim = time_dim # This is num_relation if my interpretation above is correct for the view
        self.in_dim = in_dim # This is (num_nodes * features_from_diffusion) / num_relation
        self.inter_dim = inter_dim
        self.out_dim = out_dim
        self.activation = torch.nn.PReLU()
        self.Q_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.K_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.V_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.ret_feat = torch.nn.Linear(self.inter_dim, self.out_dim)

    def forward(self, x_batched, d_gamma_batched):
        # x_batched: (Batch_Size, Num_Relations, Num_Nodes, Features_from_Diffusion)
        # d_gamma_batched: (Batch_Size, Time_Dim_Retention, Time_Dim_Retention)
        #      or d_gamma: (Time_Dim_Retention, Time_Dim_Retention) to be broadcasted/repeated.
        #      self.time_dim is Time_Dim_Retention (e.g., window_size)
        #      self.in_dim is the feature dimension for QKV layers after reshaping x_sample.

        device = x_batched.device
        batch_size = x_batched.shape[0]
        num_nodes_original = x_batched.shape[2] # Num_Nodes

        # Ensure d_gamma is on the correct device and potentially batched
        if d_gamma_batched.dim() == 2: # If a single D_gamma is passed
            d_gamma_batched = d_gamma_batched.to(device) #.unsqueeze(0).repeat(batch_size, 1, 1)
        elif d_gamma_batched.dim() == 3 and d_gamma_batched.shape[0] == batch_size:
            d_gamma_batched = d_gamma_batched.to(device)
        else:
            raise ValueError(f"d_gamma shape {d_gamma_batched.shape} is not compatible with batch_size {batch_size}")

        batch_outputs = []
        for b_idx in range(batch_size):
            # x_sample: (Num_Relations, Num_Nodes, Features_from_Diffusion)
            x_sample = x_batched[b_idx]
            
            # The critical reshape:
            # x_sample must be reshaped to (self.time_dim, self.in_dim) for QKV.
            # self.time_dim is, e.g., window_size.
            # self.in_dim is (Num_Relations * Num_Nodes * Features_from_Diffusion) / self.time_dim
            # This requires the product of dimensions of x_sample to be divisible by self.time_dim,
            # and for self.in_dim to be correctly configured during __init__.
            try:
                x_reshaped_sample = x_sample.contiguous().view(self.time_dim, self.in_dim)
            except RuntimeError as e:
                raise RuntimeError(f"Error reshaping x_sample in ParallelRetention for batch item {b_idx}. "
                                   f"x_sample shape: {x_sample.shape}, target view: ({self.time_dim}, {self.in_dim}). "
                                   f"Original error: {e}")

            q_sample = self.Q_layers(x_reshaped_sample) # (time_dim, inter_dim)
            k_sample = self.K_layers(x_reshaped_sample) # (time_dim, inter_dim)
            v_sample = self.V_layers(x_reshaped_sample) # (time_dim, inter_dim)

            inter_feat_sample = torch.matmul(q_sample, k_sample.transpose(0, 1)) # (time_dim, time_dim)
            
            current_d_gamma = d_gamma_batched if d_gamma_batched.dim() == 2 else d_gamma_batched[b_idx]

            retained_x_sample = torch.matmul(current_d_gamma * inter_feat_sample, v_sample) # (time_dim, inter_dim)
            
            output_x_sample = self.activation(self.ret_feat(retained_x_sample)) # (time_dim, out_dim)

            # Reshape to (Num_Nodes, Features_after_Retention)
            # This requires self.time_dim * self.out_dim to be divisible by num_nodes_original.
            # Features_after_Retention = (self.time_dim * self.out_dim) / num_nodes_original
            try:
                final_output_sample = output_x_sample.contiguous().view(num_nodes_original, -1)
            except RuntimeError as e:
                 raise RuntimeError(f"Error reshaping output_x_sample in ParallelRetention for batch item {b_idx}. "
                                   f"output_x_sample shape: {output_x_sample.shape}, target num_nodes: {num_nodes_original}. "
                                   f"Original error: {e}")
            batch_outputs.append(final_output_sample)

        # Stack outputs for all batch items
        # Resulting shape: (Batch_Size, Num_Nodes, Features_after_Retention)
        return torch.stack(batch_outputs, dim=0)


class MGDPR(nn.Module):
    def __init__(self, diffusion_config, retention_config, ret_linear_1_config, ret_linear_2_config, post_pro_config,
                 layers, num_nodes, time_dim, num_relation, gamma, expansion_steps):
        super(MGDPR, self).__init__()

        self.layers = layers
        self.num_nodes = num_nodes
        self.time_dim = time_dim # This is time_steps (window size)
        self.num_relation = num_relation

        self.T = nn.Parameter(torch.empty(layers, num_relation, expansion_steps, num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.T)

        self.theta = nn.Parameter(torch.empty(layers, num_relation, expansion_steps))
        nn.init.xavier_uniform_(self.theta)

        lower_tri = torch.tril(torch.ones(time_dim, time_dim), diagonal=-1)
        # D_gamma_tensor = torch.where(lower_tri == 0, torch.tensor(0.0), gamma ** -lower_tri) # Original
        # Paper uses zeta for decay in retention, gamma for diffusion constraint.
        # The D_gamma in notebook is based on `gamma` param to MGDPR.
        # Let's assume gamma is the decay coefficient for retention here as per notebook.
        # The paper's D_ij = zeta^(i-j). Here it's gamma^(-lower_tri_value).
        # If lower_tri_value is (i-j) for i>j, then gamma^-(i-j).
        # This seems consistent if gamma is zeta.
        D_gamma_tensor = torch.zeros_like(lower_tri)
        non_zero_mask = lower_tri != 0
        # Ensure gamma is positive if it's a base for exponentiation
        # Using abs(gamma) or ensuring gamma > 0 during init might be safer if gamma can be negative.
        # However, decay factors are usually > 0.
        # The paper uses zeta, typically > 1 for decay. If gamma is small (e.g., 2.5e-4), gamma^-(i-j) will be very large.
        # If gamma is the decay factor itself (e.g. 0.9), then gamma^(i-j).
        # The notebook has `gamma ** -lower_tri`. If lower_tri is 1, 2, 3... for i-j.
        # Then gamma^-1, gamma^-2, ...
        # If gamma = 2.5e-4, then (1/gamma)^1, (1/gamma)^2 ... these are large.
        # If gamma is decay like 0.9, then 0.9^-1, 0.9^-2... also large.
        # The paper's D_ij = zeta^(i-j) where zeta is decay. If zeta < 1, it decays.
        # If D_gamma is for (QK^T . D)V, D should be decaying.
        # RetNet paper: D_nm = gamma^(n-m) for n>=m. gamma is a scalar between 0 and 1.
        # So, if `lower_tri` gives `n-m` for `n>m`, then `gamma_decay^(n-m)`.
        # The `gamma` parameter to MGDPR (2.5e-4) is likely the learning rate or regularization, not retention decay.
        # The paper mentions zeta as decay coefficient for retention. This is not in MGDPR params.
        # Let's assume the notebook's D_gamma calculation is what's intended, using the MGDPR `gamma` parameter.
        # This might be an area for review based on RetNet principles if `gamma` is not the retention decay.
        # For now, replicate notebook.
        D_gamma_tensor[non_zero_mask] = gamma ** (-lower_tri[non_zero_mask])
        self.register_buffer('D_gamma', D_gamma_tensor)


        self.diffusion_layers = nn.ModuleList(
            # diffusion_config is [in0, out0, in1, out1, ...]
            # Each MultiReDiffusion layer corresponds to an MGDPR block.
            # The number of such blocks is self.layers.
            # The number of pairs in diffusion_config should be self.layers.
            [MultiReDiffusion(diffusion_config[2*i], diffusion_config[2*i + 1], num_relation)
             for i in range(len(diffusion_config) // 2)]
        )
        # Ensure the number of created diffusion layers matches self.layers
        if len(self.diffusion_layers) != self.layers:
            raise ValueError(f"Mismatch between number of MGDPR layers ({self.layers}) and "
                             f"diffusion_layers created ({len(self.diffusion_layers)}) from diffusion_config. "
                             f"Expected diffusion_config to have {self.layers * 2} elements.")

        # retention_config is a flat list: [in0, inter0, out0, in1, inter1, out1, ...]
        self.retention_layers = nn.ModuleList(
            [ParallelRetention(time_dim, retention_config[3 * i], retention_config[3 * i + 1], retention_config[3 * i + 2])
             for i in range(len(retention_config) // 3)]
        )
        
        # ret_linear_1_config is flat: [in0, out0, in1, out1, ...]
        self.ret_linear_1 = nn.ModuleList(
            [nn.Linear(ret_linear_1_config[2 * i], ret_linear_1_config[2 * i + 1])
             for i in range(len(ret_linear_1_config) // 2)]
        )

        self.ret_linear_2 = nn.ModuleList(
            [nn.Linear(ret_linear_2_config[2 * i], ret_linear_2_config[2 * i + 1])
             for i in range(len(ret_linear_2_config) // 2)]
        )

        self.mlp = nn.ModuleList(
            [nn.Linear(post_pro_config[i], post_pro_config[i + 1]) for i in range(len(post_pro_config) - 1)]
        )
        self.activation_mlp = nn.PReLU() # Assuming PReLU based on other activations, or could be configurable

    def forward(self, x_batch, a_batch):
        # x_batch: (Batch_Size, Num_Relations, Num_Nodes, Features_Initial)
        # a_batch: (Batch_Size, Num_Relations, Num_Nodes, Num_Nodes)
        
        device = x_batch.device
        batch_size = x_batch.shape[0]

        # h_for_diffusion is the input to the diffusion part of each MGDPR layer.
        # Initialized with x_batch. Shape: (B, R, N, F_in)
        h_for_diffusion = x_batch.to(device)
        
        # h_prime_retained_for_skip is the output of the retention block (after ret_linear_2) from the *previous* MGDPR layer.
        # This is used for the skip connection in the current MGDPR layer's retention block.
        # Initialized to None, special handling for l_layer_idx == 0.
        # Shape: (B, N, F_from_prev_rl2)
        h_prime_retained_for_skip = None

        # This will store the final output of the retention block for the current layer, to be used by MLP.
        # Shape: (B, N, F_mlp_in_features)
        final_layer_output_for_mlp = None

        for l_layer_idx in range(self.layers):
            # --- Multi-relational Graph Diffusion ---
            # Input h_for_diffusion: (B, R, N, F_current_diffusion_input)
            # Output h_diffused, u_intermediate: (B, R, N, F_diffusion_output)
            h_diffused, u_intermediate = self.diffusion_layers[l_layer_idx](
                self.theta[l_layer_idx],  # (R, ExpSteps)
                self.T[l_layer_idx],      # (R, ExpSteps, N, N)
                a_batch,                  # (B, R, N, N)
                h_for_diffusion           # (B, R, N, F_current_diffusion_input)
            )
            
            # Update h_for_diffusion for the *next* MGDPR layer's diffusion part.
            h_for_diffusion = h_diffused
            u_intermediate = u_intermediate.to(device) # (B, R, N, F_diffusion_output)

            # --- Parallel Retention ---
            # Input u_intermediate: (B, R, N, F_diffusion_output)
            # Input self.D_gamma: (Time_Dim_Ret, Time_Dim_Ret) - not batched, handled by ParallelRetention
            # Output eta_batch: (B, N, F_retention_output)
            eta_batch = self.retention_layers[l_layer_idx](u_intermediate, self.D_gamma)
            eta_batch = eta_batch.to(device)

            # --- Decoupled Representation Transform (Skip Connections) ---
            if l_layer_idx == 0:
                # For the first layer, the skip connection comes from the initial input x_batch.
                # x_batch: (B, R, N, F_initial)
                # Reshape for ret_linear_1: (B, N, R * F_initial)
                if x_batch.dim() == 4: # (B, R, N, F_initial)
                    # Permute to (B, N, R, F_initial) then view as (B, N, R * F_initial)
                    skip_connection_source = x_batch.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_nodes, -1)
                else:
                    raise ValueError(f"Initial x_batch has unexpected dimension: {x_batch.dim()}, expected 4D (B,R,N,F)")
            else:
                # For subsequent layers, the skip connection comes from the previous layer's h_prime_retained_for_skip.
                # h_prime_retained_for_skip: (B, N, F_from_prev_rl2)
                skip_connection_source = h_prime_retained_for_skip
            
            # transformed_skip: (B, N, F_rl1_output)
            transformed_skip = self.ret_linear_1[l_layer_idx](skip_connection_source)
            
            # h_concat: (B, N, F_eta + F_rl1_output)
            h_concat = torch.cat((eta_batch, transformed_skip), dim=2) # Concatenate along the feature dimension
            
            # current_h_prime_retained: (B, N, F_rl2_output)
            current_h_prime_retained = self.ret_linear_2[l_layer_idx](h_concat)
            
            # This output becomes the skip connection source for the next layer.
            h_prime_retained_for_skip = current_h_prime_retained
            
            # If this is the last MGDPR layer, its output is what goes to the MLP.
            if l_layer_idx == self.layers - 1:
                final_layer_output_for_mlp = current_h_prime_retained

        # --- MLP Post-processing ---
        # final_layer_output_for_mlp should be (B, N, F_mlp_input)
        if final_layer_output_for_mlp is None:
             # This case should ideally not happen if self.layers > 0
            if self.layers == 0: # If no MGDPR layers, MLP processes initial x transformed.
                if x_batch.dim() == 4:
                    current_rep_for_mlp = x_batch.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_nodes, -1)
                else: # Should not happen with current DataLoader
                    raise ValueError("x_batch has unexpected shape for MLP input when layers=0")
            else: # Should not be reached if loop ran
                raise ValueError("final_layer_output_for_mlp is None after MGDPR layers.")
        else:
            current_rep_for_mlp = final_layer_output_for_mlp

        for i, mlp_layer in enumerate(self.mlp):
            current_rep_for_mlp = mlp_layer(current_rep_for_mlp)
            if i < len(self.mlp) - 1: # Apply activation to all but the last MLP layer
                current_rep_for_mlp = self.activation_mlp(current_rep_for_mlp)
        
        # Output: (Batch_Size, Num_Nodes, Num_Classes)
        return current_rep_for_mlp

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.T)
        nn.init.xavier_uniform_(self.theta)
        for module in self.modules():
            if hasattr(module, 'reset_parameters') and module is not self:
                 # Check if it's an nn.Module and not the MGDPR itself or a container like ModuleList
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d): # More specific
                    module.reset_parameters()
                elif isinstance(module, MultiReDiffusion) or isinstance(module, ParallelRetention):
                    # These don't have a standard reset_parameters, rely on their own init or MGDPR's T/theta
                    pass
