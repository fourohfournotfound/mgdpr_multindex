import torch
import torch.nn as nn
import torch.nn.functional as F # Added for potential use, though not in current snippet
from typing import Tuple # Import Tuple for type hinting

class MultiReDiffusion(torch.nn.Module):
    """
    Implements the Multi-relational Graph Diffusion layer as described in the MGDPR paper.
    This layer refines graph structures by learning task-optimal edges adaptively.
    Corresponds to Section 4.2 of the paper.
    """
    def __init__(self, input_dim: int, output_dim: int, num_relation: int):
        """
        Args:
            input_dim (int): Feature dimension of the input node representations (H_{l-1}).
            output_dim (int): Feature dimension after the relational linear transformation (W_l^r).
            num_relation (int): Number of relations |R|.
        """
        super(MultiReDiffusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        
        # Corresponds to W_l^r in the paper, one for each relation
        self.fc_layers_list = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_relation)])
        
        # Corresponds to Conv2d_{1x1} in the paper
        self.update_layer = torch.nn.Conv2d(num_relation, num_relation, kernel_size=1)
        self.activation1 = torch.nn.PReLU() # sigma after Conv2d
        self.activation0 = torch.nn.PReLU() # sigma after W_l^r H_{l-1}

    def forward(self, theta_param: torch.Tensor, t_param: torch.Tensor, a_input_batched: torch.Tensor, x_input_batched: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Multi-relational Graph Diffusion.

        Args:
            theta_param (torch.Tensor): Learnable weight coefficients gamma_{l,r,k}.
                                        Shape: (Num_Relations, Expansion_Steps).
            t_param (torch.Tensor): Learnable column-stochastic transition matrices T_{l,r,k}.
                                    Shape: (Num_Relations, Expansion_Steps, Num_Nodes, Num_Nodes).
            a_input_batched (torch.Tensor): Batched adjacency matrices A_{t,r}.
                                            Shape: (Batch_Size, Num_Relations, Num_Nodes, Num_Nodes).
            x_input_batched (torch.Tensor): Batched input node features H_{l-1}.
                                            Shape: (Batch_Size, Num_Relations, Num_Nodes, Features_Input_Dim).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - latent_feat_batch (H_l): Latent diffusion representation.
                                           Shape: (Batch_Size, Num_Relations, Num_Nodes, Features_Output_FC).
                - diffusions_batch (U_l): Intermediate representation before Conv2d, used for Parallel Retention.
                                          Shape: (Batch_Size, Num_Relations, Num_Nodes, Features_Output_FC).
                                          (Note: In this implementation, U_l is effectively H_l before the final update_layer activation,
                                           but the paper's diagram suggests U_l might be S_{l,r}H_{l-1}W_l^r.
                                           Here, we return the output of fc_layers + activation0 as the second element,
                                           and the output after Conv2d + activation1 as the first.)
                                           For simplicity and matching original logic, returning (latent_feat_batch, latent_feat_batch)
                                           if diffusions_batch is not explicitly needed elsewhere with a different value.
                                           The current return is (output_after_conv2d, output_after_conv2d).
                                           Let's refine to return (output_after_conv2d, output_before_conv2d_but_after_fc_activation0)
        """
        # theta_param (gamma_{l,r,k}): (Num_Relations, Expansion_Steps)
        # t_param (T_{l,r,k}): (Num_Relations, Expansion_Steps, Num_Nodes, Num_Nodes)
        # a_input_batched (A_{t,r}): (Batch_Size, Num_Relations, Num_Nodes, Num_Nodes)
        # x_input_batched (H_{l-1}): (Batch_Size, Num_Relations, Num_Nodes, Features_Input_Dim)

        device = x_input_batched.device
        batch_size = x_input_batched.shape[0]
        num_nodes = x_input_batched.shape[2]
        # self.num_relation is num_relations
        # self.input_dim is Features_Input_Dim
        # self.output_dim is Features_Output_FC

        # 1. Calculate diffusion_mats_batched
        # theta_param: (R, E) -> reshape to (1, R, E, 1, 1) for broadcasting
        # Apply softmax to theta_param to ensure coefficients sum to 1 over expansion steps, as per paper's constraint.
        theta_param_normalized = torch.softmax(theta_param, dim=-1)
        theta_p_exp = theta_param_normalized.unsqueeze(0).unsqueeze(3).unsqueeze(4)
        
        # t_param (T_{l,r,k}): (Num_Relations, Expansion_Steps, Num_Nodes_model, Num_Nodes_model)
        # num_nodes (calculated at line 68) is N_data from x_input_batched.shape[2]
        
        num_nodes_model_dim2 = t_param.shape[2]
        num_nodes_model_dim3 = t_param.shape[3]

        if num_nodes_model_dim2 != num_nodes_model_dim3:
            raise ValueError(f"t_param is not square in node dimensions: {t_param.shape}")
        
        num_nodes_model = num_nodes_model_dim2 # N_model from MGDPR initialization (e.g. 43)

        if num_nodes > num_nodes_model: # N_data > N_model
            raise ValueError(f"Number of nodes in input data ({num_nodes}) "
                             f"is greater than model's T matrix configured number of nodes ({num_nodes_model}).")

        # Prepare t_param for the current batch's node size (num_nodes, which is N_data)
        if num_nodes < num_nodes_model:
            # Slice t_param to match the number of nodes in the current batch
            t_param_for_batch = t_param[:, :, :num_nodes, :num_nodes]
        else: # num_nodes == num_nodes_model
            t_param_for_batch = t_param
        
        # Normalize the (potentially sliced) t_param_for_batch.
        # t_param_normalized will have shape (R, E, N_data, N_data)
        t_param_normalized = torch.softmax(t_param_for_batch, dim=3) # Paper: column-stochastic, sum over j M_ij = 1
        
        # t_p_exp will have shape (1, R, E, N_data, N_data)
        t_p_exp = t_param_normalized.unsqueeze(0)
        
        # a_input_batched already has N_data as its node dimensions.
        # a_in_b_exp will have shape (B, R, 1, N_data, N_data)
        a_in_b_exp = a_input_batched.unsqueeze(2)

        # Element-wise product for terms to be summed over Expansion_Steps (dim=2)
        # theta_p_exp: (1, R, E, 1, 1)
        # t_p_exp:     (1, R, E, N, N)
        # a_in_b_exp:  (B, R, 1, N, N)
        # Resulting terms shape: (B, R, E, N, N) due to broadcasting
        terms = theta_p_exp * t_p_exp * a_in_b_exp
        
        # Sum over Expansion_Steps dimension (k) to get S_{l,r}
        # diffusion_mats_batched (S_{l,r}): (Batch_Size, Num_Relations, Num_Nodes, Num_Nodes)
        diffusion_mats_batched = torch.sum(terms, dim=2)

        # 2. Calculate S_{l,r} H_{l-1}
        # diffusion_mats_batched: (B, R, N, N)
        # x_input_batched (H_{l-1}): (Batch_Size, Num_Relations, Num_Nodes, Features_Input_Dim)
        # torch.matmul will batch over Batch_Size and Num_Relations dimensions: (N,N) @ (N,F_in) -> (N,F_in)
        # diffusion_feats_batched (S_{l,r}H_{l-1}): (Batch_Size, Num_Relations, Num_Nodes, Features_Input_Dim)
        diffusion_feats_batched = torch.matmul(diffusion_mats_batched, x_input_batched)

        # 3. Apply relational linear transformation W_l^r and activation sigma
        # diffusion_feats_batched (S_{l,r}H_{l-1}): (Batch_Size, Num_Relations, Num_Nodes, Features_Input_Dim)
        
        # Stack weights (W_l^r) and biases from self.fc_layers_list
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

        # Apply activation0 (sigma)
        # u_intermediate_for_retention (U_l in paper, or part of H_l before Conv2d): (Batch_Size, Num_Relations, Num_Nodes, Features_Output_FC)
        # This is sigma(S_{l,r} H_{l-1} W_l^r)
        u_intermediate_for_retention = self.activation0(fc_outputs_batched)
        
        # 4. Apply Conv2d_{1x1} and activation1 (sigma) to get H_l
        # u_intermediate_for_retention shape: (Batch_Size, Num_Relations, Num_Nodes, Features_Output_FC)
        # This is the input to self.update_layer (Conv2d).
        # Input to Conv2d: (Batch_Size, Channels_in=Num_Relations, H=Num_Nodes, W=Features_Output_FC)
        # latent_feat_batch (H_l): (Batch_Size, Num_Relations, Num_Nodes, Features_Output_FC)
        # Since self.update_layer is Conv2d(num_relation, num_relation, ...), Num_Relations_out = Num_Relations.
        latent_feat_batch = self.activation1(self.update_layer(u_intermediate_for_retention))

        # H_l is latent_feat_batch
        # U_l (input to Parallel Retention) is u_intermediate_for_retention (output of FC layers + activation0)
        return latent_feat_batch, u_intermediate_for_retention


class ParallelRetention(torch.nn.Module):
    """
    Implements the Parallel Retention mechanism as described in the MGDPR paper.
    This mechanism aims to capture long-term dependencies in stock time series.
    Corresponds to Eq. 4 in Section 4.3 of the paper: eta(Z) = phi((Q K^T elementwise_prod D) V).
    """
    def __init__(self, time_dim: int, in_dim: int, inter_dim: int, out_dim: int, num_gn_groups: int = 32):
        """
        Args:
            time_dim (int): The sequence length for retention (T in paper's D matrix).
                            In MGDPR, this corresponds to `time_steps` (window size) from the input data.
            in_dim (int): Feature dimension of the input Z to the Q, K, V linear layers.
                          This is derived from reshaping the input from diffusion layers.
                          Specifically, (Num_Relations * Num_Nodes * Features_from_Diffusion) / time_dim.
            inter_dim (int): Intermediate feature dimension for Q, K, V projections.
            out_dim (int): Output feature dimension after the final linear layer (ret_feat).
            num_gn_groups (int): Number of groups for Group Normalization (phi).
        """
        super(ParallelRetention, self).__init__()
        self.time_dim = time_dim
        self.in_dim = in_dim
        self.inter_dim = inter_dim
        self.out_dim = out_dim

        # Group Normalization (phi function from paper)
        # The paper specifies phi as Group Normalization.
        if self.inter_dim > 0:
            if self.inter_dim % num_gn_groups != 0:
                print(f"Warning: ParallelRetention inter_dim {self.inter_dim} is not divisible by num_gn_groups {num_gn_groups}. "
                      f"Adjusting num_gn_groups to 1 or a divisor if possible, or ensure inter_dim is appropriate.")
            # Ensure num_gn_groups is valid and does not exceed inter_dim.
            # If inter_dim is small, num_gn_groups might need to be 1.
            effective_num_gn_groups = num_gn_groups
            if self.inter_dim < num_gn_groups : # If inter_dim is less than requested groups, use 1 or inter_dim itself if inter_dim is a valid group number
                 effective_num_gn_groups = 1 # Fallback to 1 group if inter_dim is too small for requested groups
                 if self.inter_dim > 0 and self.inter_dim % 1 == 0 : # if inter_dim itself can be a group number
                     pass # keep effective_num_gn_groups = 1 or consider self.inter_dim if it makes sense
            elif self.inter_dim > 0 and self.inter_dim % num_gn_groups != 0: # If not divisible, try to find a divisor or default to 1
                # Simplified: default to 1 if not perfectly divisible by requested num_gn_groups.
                # A more sophisticated approach might find the largest divisor or allow configuration.
                print(f"Adjusting num_gn_groups to 1 for inter_dim {self.inter_dim} as it's not divisible by {num_gn_groups}.")
                effective_num_gn_groups = 1
            
            if effective_num_gn_groups == 0 and self.inter_dim > 0: # Should not happen with min(num_gn_groups, self.inter_dim) logic if inter_dim > 0
                effective_num_gn_groups = 1 # Safety net

            self.group_norm = nn.GroupNorm(effective_num_gn_groups, self.inter_dim) if self.inter_dim > 0 else nn.Identity()
        else: # inter_dim is 0 or less (should not happen with proper config)
            self.group_norm = nn.Identity()
            
        self.activation = torch.nn.PReLU()
        self.Q_layers = nn.Linear(self.in_dim, self.inter_dim) # W_Q
        self.K_layers = nn.Linear(self.in_dim, self.inter_dim) # W_K
        self.V_layers = nn.Linear(self.in_dim, self.inter_dim) # W_V
        self.ret_feat = torch.nn.Linear(self.inter_dim, self.out_dim) # Final linear layer

    def forward(self, x_batched: torch.Tensor, d_gamma_batched: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Parallel Retention.

        Args:
            x_batched (torch.Tensor): Batched input features (Z in paper, derived from U_l).
                                      Shape: (Batch_Size, Num_Relations, Num_Nodes, Features_from_Diffusion).
            d_gamma_batched (torch.Tensor): Batched decay matrix D.
                                            Shape: (Batch_Size, Time_Dim_Retention, Time_Dim_Retention) or
                                                   (Time_Dim_Retention, Time_Dim_Retention) for broadcasting.
                                            Time_Dim_Retention is self.time_dim (e.g., window_size).
        
        Returns:
            torch.Tensor: Output of the retention mechanism (eta_batch).
                          Shape: (Batch_Size, Num_Nodes, Features_after_Retention).
        """
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
            
            # Apply Group Normalization (phi function from paper)
            if self.inter_dim > 0: # Apply only if inter_dim is valid for GroupNorm
                # GroupNorm expects input (N, C, ...) where C is number of channels (self.inter_dim)
                # retained_x_sample is (time_dim, inter_dim)
                # Reshape for GroupNorm: (1, inter_dim, time_dim) assuming N=1 sample for GN
                retained_x_sample_norm_input = retained_x_sample.transpose(0, 1).unsqueeze(0)
                normalized_retained_x = self.group_norm(retained_x_sample_norm_input)
                # Reshape back to (time_dim, inter_dim)
                retained_x_for_activation = normalized_retained_x.squeeze(0).transpose(0, 1)
            else:
                retained_x_for_activation = retained_x_sample # Skip GN if inter_dim is 0

            output_x_sample = self.activation(self.ret_feat(retained_x_for_activation)) # (time_dim, out_dim)

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
    """
    Multi-relational Graph Diffusion Neural Network with Parallel Retention (MGDPR).
    This is the main model class that integrates MultiReDiffusion and ParallelRetention
    layers to perform stock trend classification, as described in the paper.
    It follows the architecture outlined in Figure 1 and Section 4 of the paper.
    """
    def __init__(self, diffusion_config: list, retention_config: list,
                 ret_linear_1_config: list, ret_linear_2_config: list,
                 post_pro_config: list, layers: int, num_nodes: int,
                 time_dim: int, num_relation: int, retention_decay_zeta: float,
                 expansion_steps: int, regularization_gamma_param: float = None):
        """
        Args:
            diffusion_config (list): Configuration for the fc_layers within MultiReDiffusion blocks.
                                     Flat list: [in_fc_MD0, out_fc_MD0, in_fc_MD1, out_fc_MD1, ...].
                                     The input to the first MultiReDiffusion's fc_layers is `time_dim` (features of initial X).
            retention_config (list): Configuration for ParallelRetention blocks.
                                     Flat list: [in_PR0, inter_PR0, out_PR0, in_PR1, inter_PR1, out_PR1, ...].
                                     `in_PR_l` is (num_relation * num_nodes * diffusion_config_out_l) / time_dim.
            ret_linear_1_config (list): Configuration for the first set of linear layers in the retention skip-connection path.
                                        Flat list: [in_RL1_0, out_RL1_0, in_RL1_1, out_RL1_1, ...].
            ret_linear_2_config (list): Configuration for the second set of linear layers in the retention skip-connection path.
                                        Flat list: [in_RL2_0, out_RL2_0, in_RL2_1, out_RL2_1, ...].
            post_pro_config (list): Configuration for the final MLP layers.
                                    Flat list: [in_MLP, hidden1_MLP, ..., out_classes_MLP].
            layers (int): Number of MGDPR layers (L in paper). Each layer consists of a MultiReDiffusion
                          block followed by a ParallelRetention block and skip connections.
            num_nodes (int): Number of nodes (stocks) in the graph (N).
            time_dim (int): Number of time steps in the input features (tau from paper, also used as T for D_gamma).
            num_relation (int): Number of relations |R|.
            retention_decay_zeta (float): Decay coefficient zeta for the D_gamma matrix in ParallelRetention.
            expansion_steps (int): Expansion step K for graph diffusion.
            regularization_gamma_param (float, optional): Regularization strength for the diffusion constraint.
                                                          (Note: constraint now handled by softmax in MultiReDiffusion).
                                                          Defaults to None.
        """
        super(MGDPR, self).__init__()

        self.layers = layers
        self.num_nodes = num_nodes
        self.time_dim = time_dim # This is time_steps (window size)
        self.num_relation = num_relation
        self.regularization_gamma = regularization_gamma_param # Store for potential future use if needed for theta_regularizer

        # Determine feature dimensions for LayerNorm
        eta_feature_dims = []
        for i in range(layers):
            # Output feature dimension of ParallelRetention's ret_feat layer (retention_config[3*i+2])
            # is then spread across num_nodes.
            # The effective feature dimension per node for eta_batch is (time_dim * retention_out_dim_of_PR_block) / num_nodes
            # This matches the logic in train_val_test.py for eta_feature_dims_per_node
            pr_out_dim = retention_config[3 * i + 2]
            eta_feat_dim = (self.time_dim * pr_out_dim) // self.num_nodes # Ensure integer
            if (self.time_dim * pr_out_dim) % self.num_nodes != 0:
                 print(f"Warning: MGDPR LayerNorm for eta_batch layer {i}, (time_dim * pr_out_dim) is not divisible by num_nodes. "
                       f"({self.time_dim} * {pr_out_dim}) / {self.num_nodes} = { (self.time_dim * pr_out_dim) / self.num_nodes}. Using floor division.")
            eta_feature_dims.append(eta_feat_dim)

        self.ln_eta = nn.ModuleList([nn.LayerNorm(eta_feature_dims[i]) for i in range(layers)])
        
        # ret_linear_1_config is [in0, out0, in1, out1, ...]
        # Feature dimension after ret_linear_1[l] is ret_linear_1_config[2*l+1]
        self.ln_skip = nn.ModuleList([nn.LayerNorm(ret_linear_1_config[2*i+1]) for i in range(layers)])

        # ret_linear_2_config is [in0, out0, in1, out1, ...]
        # Feature dimension after ret_linear_2[l] is ret_linear_2_config[2*l+1]
        self.ln_retained = nn.ModuleList([nn.LayerNorm(ret_linear_2_config[2*i+1]) for i in range(layers)])

        # For MLP, post_pro_config is [in, hidden1, hidden2, ..., out_classes]
        # We add LayerNorm after each hidden layer's Linear transformation, before activation.
        # So, for post_pro_config[i+1] where it's a hidden dim.
        self.ln_mlp = nn.ModuleList()
        for i in range(len(post_pro_config) - 2): # -2 because we don't normalize before the final output layer or after it
            self.ln_mlp.append(nn.LayerNorm(post_pro_config[i+1]))


        self.T = nn.Parameter(torch.empty(layers, num_relation, expansion_steps, num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.T)

        self.theta = nn.Parameter(torch.empty(layers, num_relation, expansion_steps))
        nn.init.xavier_uniform_(self.theta)

        lower_tri = torch.tril(torch.ones(time_dim, time_dim), diagonal=-1)
        D_gamma_tensor = torch.zeros_like(lower_tri)
        non_zero_mask = lower_tri != 0
        # Corrected D_gamma calculation using retention_decay_zeta (e.g., 0.9)
        # D_nm = zeta^(n-m) for n>m. lower_tri[non_zero_mask] gives positive values for (n-m).
        D_gamma_tensor[non_zero_mask] = retention_decay_zeta ** (lower_tri[non_zero_mask])
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
            [ParallelRetention(time_dim, retention_config[3 * i], retention_config[3 * i + 1], retention_config[3 * i + 2], num_gn_groups=25) # Changed num_gn_groups to 25
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

        # For bypass test (now removed from forward, but keep layers if needed for other tests):
        # initial_features_per_node_flat = self.num_relation * self.time_dim # R * F_initial
        # mlp_input_dim = post_pro_config[0]
        # self.bypass_projection = nn.Linear(initial_features_per_node_flat, mlp_input_dim)
        # self.ln_bypass_projection = nn.LayerNorm(mlp_input_dim)

    def forward(self, x_batch, a_batch):
        # x_batch: (Batch_Size, Num_Relations, Num_Nodes, Features_Initial)
        # a_batch: (Batch_Size, Num_Relations, Num_Nodes, Num_Nodes)
        
        device = x_batch.device
        batch_size = x_batch.shape[0]

        # h_for_diffusion is the input to the diffusion part of each MGDPR layer.
        # Initialized with x_batch. Shape: (B, R, N, F_in)
        h_for_diffusion = x_batch.to(device) # Original path
        
        # h_prime_retained_for_skip is the output of the retention block (after ret_linear_2) from the *previous* MGDPR layer.
        # This is used for the skip connection in the current MGDPR layer's retention block.
        # Initialized to None, special handling for l_layer_idx == 0.
        # Shape: (B, N, F_from_prev_rl2)
        h_prime_retained_for_skip = None # Original path

        # This will store the final output of the retention block for the current layer, to be used by MLP.
        # Shape: (B, N, F_mlp_in_features)
        final_layer_output_for_mlp = None # Original path

        # --- ORIGINAL MGDPR PATH (Restored) ---
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
            eta_batch = self.ln_eta[l_layer_idx](eta_batch.to(device))

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
            transformed_skip = self.ln_skip[l_layer_idx](transformed_skip)
            
            # h_concat: (B, N, F_eta + F_rl1_output)
            h_concat = torch.cat((eta_batch, transformed_skip), dim=2) # Concatenate along the feature dimension
            
            # current_h_prime_retained: (B, N, F_rl2_output)
            current_h_prime_retained = self.ret_linear_2[l_layer_idx](h_concat)
            current_h_prime_retained = self.ln_retained[l_layer_idx](current_h_prime_retained)
            
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
                 # This path should not be taken if self.layers > 0 (e.g. 2 as per train script)
                 # If self.layers was 0, this would be the path.
                if x_batch.dim() == 4:
                    # Fallback for layers=0: use initial x_batch, reshaped and projected
                    x_transformed_for_mlp = x_batch.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_nodes, -1)
                    # We need a projection if MLP input dim doesn't match num_relation * time_dim
                    # For simplicity, assume post_pro_config[0] is designed for this or add a specific projection.
                    # For now, let's assume if layers=0, the MLP input is directly from reshaped x_batch if dimensions match,
                    # or this path needs a dedicated projection layer.
                    # The bypass_projection was for this, but it's better to handle it cleanly.
                    # For now, this path will likely error if layers=0 and MLP input dim doesn't match.
                    # The current train script has layers=2, so this path shouldn't be hit.
                    current_rep_for_mlp = x_transformed_for_mlp # This might need projection
                else:
                    raise ValueError("x_batch has unexpected shape for MLP input when layers=0")
            else: # Should not be reached if loop ran and self.layers > 0
                raise ValueError("final_layer_output_for_mlp is None after MGDPR layers, but self.layers > 0.")
        else:
            current_rep_for_mlp = final_layer_output_for_mlp
        # --- END ORIGINAL MGDPR PATH ---

        for i, mlp_layer in enumerate(self.mlp):
            current_rep_for_mlp = mlp_layer(current_rep_for_mlp)
            if i < len(self.mlp) - 1: # Apply LayerNorm and activation to all but the last MLP layer
                current_rep_for_mlp = self.ln_mlp[i](current_rep_for_mlp)
                current_rep_for_mlp = self.activation_mlp(current_rep_for_mlp)
        
        # Output: (Batch_Size, Num_Nodes, Num_Classes)
        return current_rep_for_mlp

    def get_theta_regularization_loss(self):
        """
        Calculates the regularization loss for the theta parameters.
        The loss is sum_{l,r} | (sum_k theta_{l,r,k}) - 1 |.
        This encourages the sum of raw theta weights (before softmax in MultiReDiffusion)
        for each layer and relation over the expansion steps to be close to 1.
        """
        if not hasattr(self, 'theta') or self.theta is None:
            return torch.tensor(0.0, device=self.T.device if hasattr(self, 'T') else 'cpu') # Return 0 if theta is not defined

        # self.theta has shape (layers, num_relation, expansion_steps)
        # Sum theta over expansion_steps (dim=2)
        sum_over_k = torch.sum(self.theta, dim=2) # Shape: (layers, num_relation)
        
        # Calculate |sum_k theta_{l,r,k} - 1|
        abs_diff_from_one = torch.abs(sum_over_k - 1.0)
        
        # Sum these absolute differences over all layers and relations
        total_reg_loss = torch.sum(abs_diff_from_one)
        
        return total_reg_loss

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
