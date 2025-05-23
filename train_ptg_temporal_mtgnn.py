import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse

# Assuming MyDataset is in this path relative to this script
# Adjust if MyDataset is located elsewhere or if this script is placed differently
from mgdpr_paper.MGDPR_paper.dataset.graph_dataset_gen import MyDataset
from pytorch_geometric_temporal.nn.recurrent import MTGNN

def main(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    # Load company list (example, replace with actual loading mechanism)
    # This should match the comlist used when MyDataset generated the .pt files
    # For simplicity, using a placeholder. In a real scenario, load from file or config.
    if args.market == "NASDAQ": # Example comlist
        # This is just a placeholder, ensure it matches your actual comlist for the dataset
        comlist = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA'] # Example
        if args.num_nodes_override:
            print(f"Warning: num_nodes_override ({args.num_nodes_override}) is set, but comlist for NASDAQ is also defined. Ensure consistency.")
            num_nodes = args.num_nodes_override
        else:
            num_nodes = len(comlist)
    elif args.comlist_path:
        with open(args.comlist_path, 'r') as f:
            comlist = [line.strip() for line in f if line.strip()]
        num_nodes = len(comlist)
    elif args.num_nodes_override:
        print(f"Warning: Using num_nodes_override ({args.num_nodes_override}) without a comlist. MyDataset might behave unexpectedly if it relies on comlist names.")
        num_nodes = args.num_nodes_override
        comlist = [f"stock_{i}" for i in range(num_nodes)] # Dummy comlist
    else:
        raise ValueError("Either --market NASDAQ (for example comlist), --comlist_path, or --num_nodes_override must be provided.")

    print(f"Number of nodes (companies): {num_nodes}")
    print(f"Comlist: {comlist[:5]}... (first 5)")


    # Dataset and DataLoader
    # Note: MyDataset's __init__ triggers graph generation if files are missing.
    # Ensure the parameters match how your .pt files were generated.
    train_dataset = MyDataset(
        root_csv_path=args.root_csv_path,
        desti=args.desti_path,
        market=args.market,
        comlist=comlist,
        start=args.train_start_date,
        end=args.train_end_date,
        window=args.window_size,
        dataset_type="Train" # Ensure this matches the subfolder name
    )

    if len(train_dataset) == 0:
        print("Training dataset is empty. Please check paths and parameters. Exiting.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True, # Shuffle for training
        collate_fn=lambda batch: {
            'X_in': torch.stack([item['X'] for item in batch]),
            'target': torch.stack([item['Y'] for item in batch]),
            'target_date_str': [item['target_date_str'] for item in batch],
            'original_ticker_order': [item['original_ticker_order'] for item in batch] # Assuming all items in batch have same order
        }
    )

    # Model instantiation
    # These are placeholders for MTGNN parameters; adjust them based on your needs/experiments.
    # in_dim corresponds to the number of features per node per time step (e.g., O,H,L,C,V -> 5)
    in_dim = args.in_dim # Should match the first dimension of 'X' in your .pt files
    # seq_length is the window_size
    seq_length = args.window_size
    # out_dim is the prediction horizon. For your current Y (next single step), this should be 1.
    out_dim = 1 # Predict one step ahead

    model = MTGNN(
        gcn_true=True,
        build_adj=True,  # Crucial: let the model learn the adjacency matrix
        gcn_depth=args.gcn_depth,
        num_nodes=num_nodes,
        kernel_set=[2, 3, 3, 2], # Example kernel set for temporal convolution
        kernel_size=7, # Example kernel size for temporal convolution
        dropout=args.dropout,
        subgraph_size=min(num_nodes, args.subgraph_size), # Ensure subgraph_size <= num_nodes
        node_dim=args.node_dim, # Dimension for node embeddings if build_adj=True
        dilation_exponential=args.dilation_exponential,
        conv_channels=args.conv_channels,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        seq_length=seq_length,
        in_dim=in_dim,
        out_dim=out_dim, # Prediction horizon
        layers=args.mtgnn_layers, # Number of MTGNN blocks
        propalpha=args.propalpha,
        tanhalpha=args.tanhalpha,
        layer_norm_affline=True, # Example
        xd=None # Static features dimension, if any
    ).to(device)

    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Loss and optimizer
    criterion = nn.MSELoss() # Example: Mean Squared Error for regression-like targets
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            X_in = batch_data['X_in'].to(device) # Shape: (batch_size, in_dim, num_nodes, seq_len)
            target_y = batch_data['target'].to(device) # Shape: (batch_size, num_nodes)

            optimizer.zero_grad()

            # MTGNN forward pass
            # A_tilde is None because build_adj=True
            # idx is None assuming no specific node permutation is needed
            # FE is None assuming no static features
            predictions = model(X_in, A_tilde=None, idx=None, FE=None)
            # Output shape: (batch_size, out_dim/pred_horizon, num_nodes, 1)
            # or (batch_size, num_nodes, out_dim/pred_horizon) if out_dim is last
            # The docs say: (batch_size, seq_len, num_nodes, 1) - here seq_len is output_seq_len
            # If out_dim (prediction horizon) is 1, output might be (batch_size, 1, num_nodes, 1)

            # Adjust prediction shape to match target_y (batch_size, num_nodes)
            # Assuming out_dim = 1, so the second dimension of predictions is 1 (prediction horizon)
            # And the last dimension is 1 (feature dim of prediction)
            if predictions.shape[1] == 1 and predictions.shape[3] == 1:
                 current_pred = predictions.squeeze(3).squeeze(1) # (batch_size, num_nodes)
            else:
                # This case needs verification based on actual MTGNN output for out_dim=1
                print(f"Unexpected prediction shape: {predictions.shape}. Adjusting, but verify.")
                current_pred = predictions[:, 0, :, 0] # Take first step of prediction horizon

            # Handle NaNs in targets if any (e.g., by masking the loss)
            # Your C_labels can have NaNs.
            nan_mask = ~torch.isnan(target_y)
            if nan_mask.sum() == 0: # All targets are NaN for this batch
                print(f"Skipping batch {batch_idx+1} in epoch {epoch+1} due to all NaN targets.")
                continue

            loss = criterion(current_pred[nan_mask], target_y[nan_mask])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch [{epoch+1}/{args.epochs}] completed. Average Training Loss: {avg_loss:.4f}")

        # Basic validation step (optional, add your validation_loader and logic)
        # model.eval()
        # with torch.no_grad():
        #     val_loss = 0
        #     for val_batch_data in validation_loader:
        #         # ... similar prediction and loss calculation ...
        #         pass
        # print(f"Epoch [{epoch+1}/{args.epochs}], Validation Loss: {val_loss / len(validation_loader):.4f}")

    print("Training finished.")

    # Save the model (optional)
    # torch.save(model.state_dict(), "mtgnn_pytorch_geometric_temporal.pth")
    # print("Model saved to mtgnn_pytorch_geometric_temporal.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MTGNN model from PyTorch Geometric Temporal")

    # Data parameters
    parser.add_argument('--root_csv_path', type=str, required=True, help="Path to the root CSV data file")
    parser.add_argument('--desti_path', type=str, required=True, help="Destination directory for processed .pt graph files (base for MyDataset)")
    parser.add_argument('--market', type=str, required=True, help="Market name (e.g., NASDAQ, Shortlist)")
    parser.add_argument('--comlist_path', type=str, default=None, help="Path to the comlist file (list of tickers, one per line)")
    parser.add_argument('--num_nodes_override', type=int, default=None, help="Override number of nodes (if comlist not used or for quick tests)")
    parser.add_argument('--train_start_date', type=str, default="2010-01-01", help="Train start date (YYYY-MM-DD)")
    parser.add_argument('--train_end_date', type=str, default="2018-12-31", help="Train end date (YYYY-MM-DD)")
    parser.add_argument('--window_size', type=int, default=10, help="Lookback window size (seq_length for MTGNN)")
    parser.add_argument('--in_dim', type=int, default=5, help="Number of input features per node (e.g., O,H,L,C,V)")

    # MTGNN Model parameters (using defaults or common values, tune these)
    parser.add_argument('--gcn_depth', type=int, default=2, help="Depth of GCN")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate")
    parser.add_argument('--subgraph_size', type=int, default=20, help="Subgraph size for adaptive adj matrix")
    parser.add_argument('--node_dim', type=int, default=40, help="Node embedding dimension for adaptive adj matrix")
    parser.add_argument('--dilation_exponential', type=int, default=2, help="Dilation exponential for TCN") # often 1 or 2
    parser.add_argument('--conv_channels', type=int, default=32, help="Convolution channels in TCN")
    parser.add_argument('--residual_channels', type=int, default=32, help="Residual channels in TCN")
    parser.add_argument('--skip_channels', type=int, default=64, help="Skip channels in TCN")
    parser.add_argument('--end_channels', type=int, default=128, help="End channels in TCN")
    parser.add_argument('--mtgnn_layers', type=int, default=3, help="Number of MTGNN blocks/layers")
    parser.add_argument('--propalpha', type=float, default=0.05, help="Alpha for mix-hop propagation")
    parser.add_argument('--tanhalpha', type=float, default=3.0, help="Alpha for tanh in adaptive adj matrix")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--log_interval', type=int, default=10, help="Log training status every N batches")
    parser.add_argument('--use_cuda', action='store_true', help="Use CUDA if available")

    args = parser.parse_args()
    main(args)