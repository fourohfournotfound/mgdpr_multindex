import torch
import torch.nn as nn
import torch.nn.functional as F # Added for potential use, though not in current snippet

class MultiReDiffusion(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_relation):
        super(MultiReDiffusion, self).__init__()
        self.input_dim = input_dim  # Store input_dim for FC vectorization
        self.output_dim = output_dim  # Store output_dim, was self.output
        self.num_relation = num_relation
        
        self.fc_layers_list = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_relation)])
        
        self.update_layer = torch.nn.Conv2d(num_relation, num_relation, kernel_size=1)
        self.activation1 = torch.nn.PReLU()
        self.activation0 = torch.nn.PReLU()

    def forward(self, theta_param, t_param, a_input_batched, x_input_batched):
        # theta_param: (Num_Relations, Expansion_Steps)
        # t_param: (Num_Relations, Expansion_Steps, Num_Nodes, Num_Nodes)
        # a_input_batched: (Batch_Size, Num_Relations, Num_Nodes, Num_Nodes)
        # x_input_batched: (Batch_Size, Num_Relations, Num_Nodes, Features_Input_Dim)

        device = x_input_batched.device
        batch_size = x_input_batched.shape[0]
        num_nodes = x_input_batched.shape[2]
        # self.num_relation is num_relations
        # self.input_dim is Features_Input_Dim
        # self.output_dim is Features_Output_FC

        # 1. Calculate diffusion_mats_batched
        # theta_param: (R, E) -> reshape to (1, R, E, 1, 1) for broadcasting
        theta_p_exp = theta_param.unsqueeze(0).unsqueeze(3).unsqueeze(4)
        # t_param: (R, E, N, N) -> reshape to (1, R, E, N, N) for broadcasting
        t_p_exp = t_param.unsqueeze(0)
        # a_input_batched: (B, R, N, N) -> reshape to (B, R, 1, N, N) for broadcasting
        a_in_b_exp = a_input_batched.unsqueeze(2)

        # Element-wise product for terms to be summed over Expansion_Steps (dim=2)
        # theta_p_exp: (1, R, E, 1, 1)
        # t_p_exp:     (1, R, E, N, N)
        # a_in_b_exp:  (B, R, 1, N, N)
        # Resulting terms shape: (B, R, E, N, N) due to broadcasting
        terms = theta_p_exp * t_p_exp * a_in_b_exp
        
        # Sum over Expansion_Steps dimension
        # diffusion_mats_batched: (B, R, N, N)
        diffusion_mats_batched = torch.sum(terms, dim=2)

        # 2. Calculate diffusion_feats_batched
        # diffusion_mats_batched: (B, R, N, N)
        # x_input_batched: (B, R, N, F_in) where F_in is self.input_dim
        # torch.matmul will batch over B and R dimensions: (N,N) @ (N,F_in) -> (N,F_in)
        # diffusion_feats_batched: (B, R, N, F_in)
        diffusion_feats_batched = torch.matmul(diffusion_mats_batched, x_input_batched)

        # 3. Apply FC layers (self.fc_layers_list) and activation0, vectorized over relations
        # diffusion_feats_batched: (B, R, N, F_in)
        
        # Stack weights and biases from self.fc_layers_list
        # all_fc_weights: (R, F_out, F_in)
        all_fc_weights = torch.stack([layer.weight for layer in self.fc_layers_list], dim=0)
        # all_fc_biases: (R, F_out)
        all_fc_biases = torch.stack([layer.bias for layer in self.fc_layers_list], dim=0)

        # Prepare input for batched matrix multiplication (bmm)
        # Input: diffusion_feats_batched (B, R, N, F_in)
        # Permute to (R, B, N, F_in)
        input_permuted_for_fc = diffusion_feats_batched.permute(1, 0, 2, 3)
        # Reshape to (R, B*N, F_in) to treat R as batch for bmm
        input_reshaped_for_fc = input_permuted_for_fc.contiguous().view(
            self.num_relation,
            batch_size * num_nodes,
            self.input_dim
        )

        # Prepare weights for bmm
        # all_fc_weights is (R, F_out, F_in). Transpose to (R, F_in, F_out) for matmul.
        weights_for_bmm = all_fc_weights.transpose(1, 2)
        
        # Apply linear transformation using bmm
        # input_reshaped_for_fc: (R, B*N, F_in)
        # weights_for_bmm:       (R, F_in, F_out)
        # fc_applied_bmm:        (R, B*N, F_out)
        fc_applied_bmm = torch.bmm(input_reshaped_for_fc, weights_for_bmm)
        
        # Add biases
        # all_fc_biases: (R, F_out) -> expand to (R, 1, F_out) for broadcasting
        bias_expanded_for_fc = all_fc_biases.unsqueeze(1)
        # fc_with_bias_bmm: (R, B*N, F_out)
        fc_with_bias_bmm = fc_applied_bmm + bias_expanded_for_fc
        
        # Reshape back to (B, R, N, F_out)
        # (R, B*N, F_out) -> (R, B, N, F_out)
        fc_outputs_reshaped = fc_with_bias_bmm.view(
            self.num_relation,
            batch_size,
            num_nodes,
            self.output_dim # F_out
        )
        # (R, B, N, F_out) -> (B, R, N, F_out)
        fc_outputs_batched = fc_outputs_reshaped.permute(1, 0, 2, 3)

        # Apply activation0
        # diffusions_batch: (B, R, N, F_out)
        diffusions_batch = self.activation0(fc_outputs_batched)
        
        # 4. Apply update_layer and activation1
        # diffusions_batch shape: (Batch_Size, Num_Relations, Num_Nodes, Features_Output_FC)
        # This is the expected input shape for self.update_layer (Conv2d)
        # Input: (Batch_Size, Channels_in=Num_Relations, H=Num_Nodes, W=Features_Output_FC)
        latent_feat_batch = self.activation1(self.update_layer(diffusions_batch))
        # latent_feat_batch shape: (Batch_Size, Num_Relations_out, Num_Nodes, Features_Output_FC)
        # Since self.update_layer is Conv2d(num_relation, num_relation, ...), Num_Relations_out = Num_Relations.

        # Return (h_diffused, u_intermediate)
        # As per original logic, both can be latent_feat_batch
        return latent_feat_batch, latent_feat_batch


class ParallelRetention(torch.nn.Module):
    def __init__(self, time_dim, in_dim, inter_dim, out_dim): # time_dim seems to be num_nodes from usage
        super(ParallelRetention, self).__init__()
        self.time_dim = time_dim 
        self.in_dim = in_dim 
        self.inter_dim = inter_dim
        self.out_dim = out_dim
        self.activation = torch.nn.PReLU()
        self.Q_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.K_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.V_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.ret_feat = torch.nn.Linear(self.inter_dim, self.out_dim)

    def forward(self, x_batched, d_retention_batched):
        # x_batched: (Batch_Size, Num_Relations, Num_Nodes, Features_from_Diffusion)
        # d_retention_batched: (Batch_Size, Time_Dim_Retention, Time_Dim_Retention)
        #      or d_retention: (Time_Dim_Retention, Time_Dim_Retention) to be broadcasted/repeated.
        #      self.time_dim is Time_Dim_Retention (e.g., window_size)
        #      self.in_dim is the feature dimension for QKV layers after reshaping x_sample.

        device = x_batched.device
        batch_size = x_batched.shape[0]
        num_nodes_original = x_batched.shape[2] # Num_Nodes

        # Ensure d_retention is on the correct device and potentially batched
        if d_retention_batched.dim() == 2: # If a single D_retention is passed
            d_retention_batched = d_retention_batched.to(device) #.unsqueeze(0).repeat(batch_size, 1, 1)
        elif d_retention_batched.dim() == 3 and d_retention_batched.shape[0] == batch_size:
            d_retention_batched = d_retention_batched.to(device)
        else:
            raise ValueError(f"d_retention shape {d_retention_batched.shape} is not compatible with batch_size {batch_size}")

        batch_outputs = []
        for b_idx in range(batch_size):
            # x_sample: (Num_Relations, Num_Nodes, Features_from_Diffusion)
            x_sample = x_batched[b_idx]
            
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
            
            current_d_retention = d_retention_batched if d_retention_batched.dim() == 2 else d_retention_batched[b_idx]

            retained_x_sample = torch.matmul(current_d_retention * inter_feat_sample, v_sample) # (time_dim, inter_dim)
            
            output_x_sample = self.activation(self.ret_feat(retained_x_sample)) # (time_dim, out_dim)

            try:
                final_output_sample = output_x_sample.contiguous().view(num_nodes_original, -1)
            except RuntimeError as e:
                 raise RuntimeError(f"Error reshaping output_x_sample in ParallelRetention for batch item {b_idx}. "
                                   f"output_x_sample shape: {output_x_sample.shape}, target num_nodes: {num_nodes_original}. "
                                   f"Original error: {e}")
            batch_outputs.append(final_output_sample)

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

        # lower_tri = torch.tril(torch.ones(time_dim, time_dim), diagonal=-1)
        # D_gamma_tensor = torch.zeros_like(lower_tri)
        # non_zero_mask = lower_tri != 0
        # D_gamma_tensor[non_zero_mask] = gamma ** (-lower_tri[non_zero_mask])
        # self.register_buffer('D_gamma', D_gamma_tensor)

        zeta = 1.27
        # Initialize on default device (CPU), will be moved by model.to(device)
        i_indices = torch.arange(time_dim).unsqueeze(1)
        j_indices = torch.arange(time_dim).unsqueeze(0)
        
        power_matrix = i_indices - j_indices
        
        # Using torch.full to create a tensor of zeta values, then raise to power_matrix
        D_values_base = torch.full((time_dim, time_dim), zeta, dtype=torch.float32)
        D_values = torch.pow(D_values_base, power_matrix.float())
        
        causal_mask = (i_indices >= j_indices).float()
        D_retention_tensor = D_values * causal_mask
        self.register_buffer('D_retention', D_retention_tensor)

        self.diffusion_layers = nn.ModuleList(
            [MultiReDiffusion(diffusion_config[2*i], diffusion_config[2*i + 1], num_relation)
             for i in range(len(diffusion_config) // 2)]
        )
        if len(self.diffusion_layers) != self.layers:
            raise ValueError(f"Mismatch between number of MGDPR layers ({self.layers}) and "
                             f"diffusion_layers created ({len(self.diffusion_layers)}) from diffusion_config. "
                             f"Expected diffusion_config to have {self.layers * 2} elements.")

        self.retention_layers = nn.ModuleList(
            [ParallelRetention(time_dim, retention_config[3 * i], retention_config[3 * i + 1], retention_config[3 * i + 2])
             for i in range(len(retention_config) // 3)]
        )
        
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
        self.activation_mlp = nn.PReLU()

    def forward(self, x_batch, a_batch):
        device = x_batch.device
        batch_size = x_batch.shape[0]

        h_for_diffusion = x_batch.to(device)
        h_prime_retained_for_skip = None
        final_layer_output_for_mlp = None

        for l_layer_idx in range(self.layers):
            h_diffused, u_intermediate = self.diffusion_layers[l_layer_idx](
                self.theta[l_layer_idx],
                self.T[l_layer_idx],
                a_batch,
                h_for_diffusion
            )
            
            h_for_diffusion = h_diffused
            u_intermediate = u_intermediate.to(device)

            eta_batch = self.retention_layers[l_layer_idx](u_intermediate, self.D_retention)
            eta_batch = eta_batch.to(device)

            if l_layer_idx == 0:
                if x_batch.dim() == 4:
                    skip_connection_source = x_batch.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_nodes, -1)
                else:
                    raise ValueError(f"Initial x_batch has unexpected dimension: {x_batch.dim()}, expected 4D (B,R,N,F)")
            else:
                skip_connection_source = h_prime_retained_for_skip
            
            transformed_skip = self.ret_linear_1[l_layer_idx](skip_connection_source)
            h_concat = torch.cat((eta_batch, transformed_skip), dim=2)
            current_h_prime_retained = self.ret_linear_2[l_layer_idx](h_concat)
            h_prime_retained_for_skip = current_h_prime_retained
            
            if l_layer_idx == self.layers - 1:
                final_layer_output_for_mlp = current_h_prime_retained

        if final_layer_output_for_mlp is None:
            if self.layers == 0: 
                if x_batch.dim() == 4:
                    current_rep_for_mlp = x_batch.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_nodes, -1)
                else: 
                    raise ValueError("x_batch has unexpected shape for MLP input when layers=0")
            else: 
                raise ValueError("final_layer_output_for_mlp is None after MGDPR layers.")
        else:
            current_rep_for_mlp = final_layer_output_for_mlp

        for i, mlp_layer in enumerate(self.mlp):
            current_rep_for_mlp = mlp_layer(current_rep_for_mlp)
            if i < len(self.mlp) - 1: 
                current_rep_for_mlp = self.activation_mlp(current_rep_for_mlp)
        
        return current_rep_for_mlp

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.T)
        nn.init.xavier_uniform_(self.theta)
        for module in self.modules():
            if hasattr(module, 'reset_parameters') and module is not self:
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d): 
                    module.reset_parameters()
                elif isinstance(module, MultiReDiffusion) or isinstance(module, ParallelRetention):
                    pass
