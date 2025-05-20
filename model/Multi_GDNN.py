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

    def forward(self, theta, t, a, x):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device should be handled by the parent module
        device = x.device # More robust: use the device of the input tensor
        diffusions = torch.zeros(theta.shape[0], a.shape[1], self.output).to(device)

        for rel in range(theta.shape[0]):
            diffusion_mat = torch.zeros_like(a[rel]).to(device) # Ensure diffusion_mat is on the correct device
            for step in range(theta.shape[-1]): # Assuming theta is (num_relation, expansion_steps)
                # Ensure t and a are also on the correct device, typically handled by parent model
                diffusion_mat += theta[rel][step] * t[rel][step] * a[rel] 

            diffusion_feat = torch.matmul(diffusion_mat, x[rel])
            diffusions[rel] = self.activation0(self.fc_layers[rel](diffusion_feat))

        # unsqueeze(0) adds a batch dimension for Conv2d, assuming diffusions is (num_relation, num_nodes, output_dim)
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
        latent_feat = self.activation1(self.update_layer(diffusions.unsqueeze(0))) # Add batch dim for Conv2d
        
        # Output of Conv2d will be (1, num_relation_out, num_nodes, self.output)
        # Since num_relation_out = num_relation (from Conv2d definition)
        # latent_feat is (1, num_relation, num_nodes, self.output)
        # Reshape to (num_relation, num_nodes, self.output) by removing the batch dim
        latent_feat = latent_feat.squeeze(0) # Remove batch dim
        # The original code had: latent_feat = latent_feat.reshape(self.num_relation, a.shape[1], -1)
        # a.shape[1] is num_nodes. So this is (num_relation, num_nodes, self.output)
        # This matches if the output of Conv2d is correctly squeezed.

        return latent_feat, diffusions


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

    def forward(self, x, d_gamma):
        # x is expected to be (num_relation, num_nodes, features_from_diffusion)
        # self.time_dim was passed as `time_steps` (window size) during MGDPR init.
        # self.in_dim was based on retention_layers config.
        # The original notebook code:
        # x.shape[1] is num_nodes
        # x = x.view(self.time_dim, -1)
        # This means (num_relation, num_nodes, features) is reshaped to (time_steps, calculated_dim)
        # This requires num_relation * num_nodes * features == time_steps * calculated_dim.
        # This seems to be a conceptual mismatch if `time_dim` is `time_steps` (window) and `x` is over relations.

        # Let's follow the paper: Z is (T x features), D is (T x T). T is sequence length.
        # If retention is applied across relations, T = num_relation.
        # Then x should be (num_relation, num_nodes * features_from_diffusion) before QKV.
        # Or, if retention is applied across nodes for each relation: T = num_nodes.
        # Then x should be (num_nodes, features_from_diffusion) for each relation.
        # Or, if retention is applied across time_steps (window) for each (node, relation): This is most common for time series.
        # But `u` doesn't have the original time_steps dimension explicitly after diffusion.
        # The `D_gamma` in MGDPR is (time_dim, time_dim) where time_dim is `time_steps` (window size).
        # This `D_gamma` is passed to `retention_layers[l](u, self.D_gamma)`.
        # This strongly implies that `x` (which is `u`) should be shaped like (some_batch_dim, time_steps, features)
        # for the `d_gamma` matrix to be applicable.
        # `u` is (num_relation, num_nodes, features_from_diffusion).
        # If `time_steps` is the retention sequence length, then `u` needs to be permuted/reshaped.
        # The `x.view(self.time_dim, -1)` in the notebook is problematic if `self.time_dim` is `time_steps` (window)
        # and `x` is `u` (num_relation, num_nodes, features).
        # The product of dimensions of `u` is `num_relation * num_nodes * features`.
        # This must be equal to `self.time_dim * new_feature_dim`.
        # So, `new_feature_dim = (num_relation * num_nodes * features) / self.time_dim`.
        # This `new_feature_dim` becomes `self.in_dim` for QKV layers.

        # Let's assume the notebook's view is intentional and `self.time_dim` is indeed `time_steps`.
        # And `x` (input `u`) is (num_relation, num_nodes, features_from_diffusion).
        num_node_original_from_x = x.shape[1] # This is num_nodes
        device = x.device
        d_gamma = d_gamma.to(device) # d_gamma is (time_steps, time_steps)

        # x is (num_relation, num_nodes, features_from_diffusion)
        # self.time_dim is `time_steps` (window size)
        # self.in_dim is the feature dimension for QKV layers after the view.
        # The view operation: x_viewed = x.view(self.time_dim, -1)
        # This means x is being reshaped such that its first dimension becomes `self.time_dim` (time_steps).
        # The elements of x are rearranged.
        # The total number of elements in x: x.shape[0]*x.shape[1]*x.shape[2]
        # This must be equal to self.time_dim * (elements / self.time_dim)
        # The second dim of x_viewed is (x.shape[0]*x.shape[1]*x.shape[2]) / self.time_dim. This is self.in_dim.
        
        # Example: u (x) is (5 relations, 30 nodes, 64 features). Total = 5*30*64 = 9600
        # time_steps (self.time_dim) = 21 (window size)
        # x_viewed = x.view(21, 9600/21) = x.view(21, 457.14) -> this will fail if not integer.
        # This implies that the product of (num_relation * num_nodes * features_from_diffusion)
        # must be divisible by `time_steps` (window_size). This is a strong constraint.

        # Let's re-read the paper's Fig 1. H_l (output of diffusion) goes into Parallel Retention.
        # H_l is (num_relation, num_nodes, features). D_gamma is (time_dim, time_dim) where time_dim is window size.
        # The MGDPR model's `self.D_gamma` is (time_dim, time_dim) where `time_dim` is `time_steps` (window).
        # This `D_gamma` is passed to `retention_layers[l](u, self.D_gamma)`.
        # This means the retention mechanism *must* operate along the `time_steps` dimension.
        # However, `u` (output of diffusion) is (num_relation, num_nodes, features_after_diffusion). It has lost the original time_steps dim.
        # This is a critical point. The `demo.ipynb` code for `ParallelRetention` seems to be applying retention
        # by reshaping `u` to have `time_steps` as its first dimension. This implies `time_steps` must have been implicitly
        # encoded or reconstructible from `u`'s dimensions, or there's a misunderstanding of how `u` relates to `time_steps`.

        # The `retention_layers` in MGDPR are initialized with `ParallelRetention(time_dim, retention_config...)`
        # where `time_dim` is `time_steps` (window size).
        # The `D_gamma` used is also based on this `time_steps`.
        # The input `x` to `ParallelRetention.forward` is `u` from diffusion.
        # `u` is (num_relation, num_nodes, features_from_diffusion).
        # The line `x = x.view(self.time_dim, -1)` means `u` is reshaped to `(time_steps, calculated_feature_dim)`.
        # This `calculated_feature_dim` is what `self.in_dim` (for QKV) should be.
        # The `retention_layers` config in `demo.ipynb` (lines 797-807) defines `in_dim`, `inter_dim`, `out_dim` for each retention layer.
        # e.g., `retention[0]` is `num_relation*3*n` (where n is num_nodes). This is the `in_dim` for the first retention layer.
        # So, `calculated_feature_dim` must be equal to `retention[3*i]`.
        # (num_relation * num_nodes * features_from_diffusion) / time_steps == retention_config_in_dim

        # Let's assume the reshaping in the notebook is what's intended, and the dimensions align.
        # `x` is `u` from diffusion: (num_relation, num_nodes, features_from_diffusion_output)
        # `self.time_dim` is `time_steps` (window size)
        # `self.in_dim` is the configured input dimension for QKV layers.
        
        # The view `x.view(self.time_dim, -1)` implies that the total number of elements in `x`
        # is `self.time_dim * self.in_dim`.
        # So, `x.shape[0] * x.shape[1] * x.shape[2] == self.time_dim * self.in_dim`.
        # This must hold for the view to be valid and for `self.in_dim` to be correctly used by Linear layers.

        x_reshaped = x.view(self.time_dim, self.in_dim) # This assumes the product of dims matches.

        q = self.Q_layers(x_reshaped) # (time_dim, inter_dim)
        k = self.K_layers(x_reshaped) # (time_dim, inter_dim)
        v = self.V_layers(x_reshaped) # (time_dim, inter_dim)

        # Original: inter_feat = self.Q_layers(x) @ self.K_layers(x).transpose(0, 1)
        # This implies x was (time_dim, in_dim)
        # So Q(x) is (time_dim, inter_dim), K(x).T is (inter_dim, time_dim)
        # QK^T is (time_dim, time_dim)
        inter_feat = torch.matmul(q, k.transpose(0, 1)) # (time_dim, time_dim)
        
        # d_gamma is (time_dim, time_dim)
        # (d_gamma * inter_feat) is element-wise product (time_dim, time_dim)
        # Then matmul with V_layers(x) which is `v` (time_dim, inter_dim)
        # So, (time_dim, time_dim) @ (time_dim, inter_dim) -> this is correct.
        retained_x = torch.matmul(d_gamma * inter_feat, v) # (time_dim, inter_dim)
        
        output_x = self.activation(self.ret_feat(retained_x)) # (time_dim, out_dim)

        # Original: return x.view(num_node, -1)
        # Here, num_node was x.shape[1] of the input x to forward, which is num_nodes.
        # output_x is (time_dim, out_dim). We need to reshape it to (num_nodes, something).
        # This requires time_dim * out_dim == num_nodes * something.
        # This means (time_steps * out_dim) must be divisible by num_nodes.
        # The `retention_layers` output dimension `retention[3*i+2]` is `self.out_dim`.
        # The final `eta` in MGDPR is expected to be (num_nodes, features_after_retention)
        # for concatenation with transformed h_prime or x.
        # So, `eta` should be (num_nodes, (time_steps * out_dim) / num_nodes).
        
        # This reshaping logic is quite specific and depends heavily on how dimensions are configured.
        # The paper's Fig 1 shows eta(H_l) then concatenated.
        # If H_l is (num_rel, num_nodes, feat_diff), and eta is (num_nodes, feat_ret),
        # this implies retention somehow aggregates or transforms across relations and time_steps.

        # Given the notebook's return `x.view(num_node_original_from_x, -1)`:
        return output_x.view(num_node_original_from_x, -1)


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
            [MultiReDiffusion(diffusion_config[i], diffusion_config[i + 1], num_relation)
             for i in range(len(diffusion_config) - 1)]
        )

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

    def forward(self, x, a):
        # x: input node features (batch_size, num_nodes, feature_dim_initial) or (num_relation, num_nodes, feature_dim_initial)
        #    The demo.ipynb loads X as (num_features/relations, num_nodes, time_window_size)
        #    The model's diffusion_layers expect x[rel] to be (num_nodes, features_for_that_relation)
        #    So, input x to MGDPR should be (num_relation, num_nodes, time_window_size_as_features)
        # a: adjacency tensor (num_relation, num_nodes, num_nodes)
        device = x.device
        h = x.to(device) # h is (num_relation, num_nodes, features)
        
        h_prime_retained = None # To store the output of the previous layer's retention block for skip connection

        for l_layer_idx in range(self.layers):
            # Multi-relational Graph Diffusion layer
            # diffusion_layers[l] expects (theta_l, T_l, a, h_current_input)
            # h_current_input should be (num_relation, num_nodes, features_for_diffusion_input)
            # Output h_diffused is (num_relation, num_nodes, features_from_diffusion_output)
            # Output u_intermediate is also (num_relation, num_nodes, features_from_diffusion_output) - same as h_diffused
            h_diffused, u_intermediate = self.diffusion_layers[l_layer_idx](
                self.theta[l_layer_idx], self.T[l_layer_idx], a, h
            )
            h = h_diffused # Update h for the next diffusion layer's input (if layers > 1 for diffusion part)
                           # Or h is the input to the retention part of the current MGDPR layer.

            u_intermediate = u_intermediate.to(device) # u is input to retention

            # Parallel Retention layer
            # retention_layers[l] expects (u_intermediate, D_gamma)
            # D_gamma is (time_dim, time_dim) where time_dim is window_size
            # u_intermediate is (num_relation, num_nodes, features_from_diffusion)
            # Output eta is (num_nodes, features_after_retention) due to the view in ParallelRetention
            eta = self.retention_layers[l_layer_idx](u_intermediate, self.D_gamma)
            eta = eta.to(device)

            # Decoupled representation transform
            if l_layer_idx == 0:
                # Original input x to MGDPR is (num_relation, num_nodes, initial_features_time_window)
                # We need to transform x to match eta's shape for concatenation, or transform x to (num_nodes, features)
                # The notebook code: x_reshaped = x.view(x.shape[1], -1)
                # This means x (num_relation, num_nodes, init_feat_len) -> (num_nodes, num_relation * init_feat_len)
                # This x_reshaped is then passed to ret_linear_1[l].
                # So, input to ret_linear_1[0] is (num_nodes, num_relation * init_feat_len)
                if x.dim() == 3: # Assuming x is (num_relation, num_nodes, feature_dim)
                    x_transformed_for_skip = x.permute(1,0,2).contiguous().view(self.num_nodes, -1)
                elif x.dim() == 2: # If x was already (num_nodes, features)
                     x_transformed_for_skip = x
                else:
                    raise ValueError(f"Initial x has unexpected dimension: {x.dim()}")

                transformed_skip = self.ret_linear_1[l_layer_idx](x_transformed_for_skip)
                h_concat = torch.cat((eta, transformed_skip), dim=1)
                h_prime_retained = self.ret_linear_2[l_layer_idx](h_concat)
            else:
                # h_prime_retained is from previous layer, (num_nodes, features_from_ret_linear_2)
                transformed_skip = self.ret_linear_1[l_layer_idx](h_prime_retained)
                h_concat = torch.cat((eta, transformed_skip), dim=1)
                h_prime_retained = self.ret_linear_2[l_layer_idx](h_concat)
            
            # The output of this block, h_prime_retained, becomes the input `h` for the *diffusion part* of the next MGDPR layer.
            # This means h_prime_retained (num_nodes, features) needs to be transformed back to
            # (num_relation, num_nodes, features) if the next diffusion layer expects that.
            # However, the diffusion layer's fc_layers take input_dim.
            # The MGDPR paper's Fig 1 shows H'_l is the final output of layer l.
            # And H_l (output of diffusion) is input to retention. H_0 is X.
            # H_l = Diffusion(H'_{l-1}, A). Eta_l = Retention(H_l). H'_l = Linear(concat(Eta_l, Linear(H'_{l-1})))
            # This means `h` for the next layer's diffusion should be `h_prime_retained` from current layer.
            # If diffusion expects (num_relation, num_nodes, features), then h_prime_retained needs reshaping/broadcasting.
            # The current `h` for diffusion is (num_relation, num_nodes, features).
            # `h_prime_retained` is (num_nodes, features).
            # This implies `h` for the next diffusion layer needs to be adapted from `h_prime_retained`.
            # The notebook code for MGDPR.forward has `h` (input to diffusion) being updated by `h, u = self.diffusion_layers[l](...)`.
            # Then `h_prime` (output of retention block) is calculated. This `h_prime` is NOT fed back as `h` to the next diffusion.
            # Instead, the `h` from `self.diffusion_layers[l]` seems to be the input to `self.diffusion_layers[l+1]`.
            # This means `h_prime_retained` is the progressive output that goes to MLP.
            # And `h` (output of diffusion) is what's iterated for the diffusion part across layers.

            # Let's re-check notebook MGDPR forward:
            # h = x (initial)
            # loop l:
            #   h, u = diffusion_layers[l](..., h)  <- h is updated here by diffusion output
            #   eta = retention_layers[l](u, ...)
            #   if l==0: h_prime = cat(eta, linear(x_reshaped))
            #   else: h_prime = cat(eta, linear(h_prime_from_prev_iteration)) <- h_prime is iterated for retention skip
            # The final h_prime is returned after MLP.
            # This means `h` for diffusion is indeed iterated from diffusion output.
            # And `h_prime_retained` is the result of the retention path.

            # So, the `h = h_diffused` line earlier was correct for iterating the diffusion part.
            # `h_prime_retained` is the one that gets processed by MLP at the end.
            if l_layer_idx < self.layers -1 : # If not the last layer, prepare 'h' for the next diffusion
                # The output of ret_linear_2 (h_prime_retained) is (num_nodes, features).
                # The input to diffusion (h) needs to be (num_relation, num_nodes, features).
                # This is where the "decoupled" nature might be tricky.
                # The paper says: H_l = sigma(Conv2d(Delta(S_lr H_{l-1} W_l^r))). This H_{l-1} is the input to diffusion.
                # And H'_l = sigma((eta(H_l) || (H'_{l-1} W^1_l + b^1_l)) W^2_l + b^2_l)
                # The output of layer l-1 is H'_{l-1}. This H'_{l-1} is used as input to layer l's diffusion (as H_{l-1})
                # and also in the skip connection for H'_l.
                # So, h (input to diffusion_layers[l]) should be h_prime_retained from layer l-1.
                # And x (initial features) is H'_{-1} effectively.

                # Corrected loop structure based on paper:
                # h_current_layer_input = x if l_layer_idx == 0 else h_prime_output_from_previous_layer
                # h_diffused, u_intermediate = self.diffusion_layers[l_layer_idx](..., h_current_layer_input)
                # ... calculate eta ...
                # skip_connection_input = x_transformed_for_skip if l_layer_idx == 0 else h_prime_output_from_previous_layer
                # ... calculate h_prime_retained ...
                # h_prime_output_from_previous_layer = h_prime_retained (for next iteration)

                # Let's adjust the loop variable `h` which is input to diffusion.
                # And `h_prime_retained` is the actual output of the MGDPR layer `l`.
                pass # h_prime_retained will be used by MLP after the loop.
                     # The `h` for diffusion needs to be managed carefully.

        # The loop in the notebook implies `h` (for diffusion) and `h_prime` (for retention output) evolve somewhat separately,
        # with `h_prime` using the *original* `x` or the *previous* `h_prime` for its skip.
        # The `h` for diffusion is taken from the output of the *previous diffusion*.
        # This means the `h = h_diffused` line was key.
        # The final `h_prime_retained` after all layers is what goes to MLP.

        final_representation = h_prime_retained # This is (num_nodes, features)
        
        for mlp_layer in self.mlp:
            final_representation = self.activation_mlp(mlp_layer(final_representation)) # Added activation
            # The last layer of MLP might not need activation if it's for logits (e.g. CrossEntropyLoss)
            # If last MLP layer is `nn.Linear(..., num_classes)`, then no activation here.
            # The demo notebook uses CrossEntropyLoss, so last MLP output should be logits.
            # Let's apply activation only for hidden MLP layers.
        
        # Refined MLP processing:
        current_rep = h_prime_retained
        for i, mlp_layer in enumerate(self.mlp):
            current_rep = mlp_layer(current_rep)
            if i < len(self.mlp) - 1: # Apply activation to all but the last MLP layer
                current_rep = self.activation_mlp(current_rep)
        
        return current_rep # (num_nodes, num_classes)

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
