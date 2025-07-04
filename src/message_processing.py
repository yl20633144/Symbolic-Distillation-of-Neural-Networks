import pandas as pd
import torch
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.linear_model import LinearRegression

def measure_latent(model, loader, device, dim, msg_dim):
    """
    Measure the model's latent message representations from a DataLoader.

    Parameters:
        model: A trained graph neural network model that has a .msg_fnc method.
        loader: A torch_geometric DataLoader (e.g., trainloader or testloader).
        device: The device on which the model and data should be placed ('cuda' or 'cpu').
        dim: Spatial dimensionality of the system (e.g., 2 for 2D, 3 for 3D).
        n_f: Feature dimension per node (e.g., position, velocity, mass/charge, etc.).
        msg_dim: Dimension of the message vector produced by the model.

    Returns:
        A pandas.DataFrame where each row corresponds to a directed edge and includes:
            - Source node features (e.g., x1, y1, vx1, vy1, q1, m1.)
            - Target node features (e.g., x2, y2, vx2, vy2, q2, m2.)
            - Message vector components (msg0, msg1, ..., m{msg_dim-1})
            - Relative displacement components dx, dy (and dz if dim == 3)
            - Distance r between nodes
    """
    model.eval()
    all_rows = []

    for batch in loader:
        # Move batch data to the specified device
        batch = batch.to(device)
        x = batch.x                # Node features, shape [N_total, n_f]
        edge_index = batch.edge_index  # Edge indices, shape [2, E]

        with torch.no_grad():
            # Extract source and target node features for each edge
            s1 = x[edge_index[0]]  # Source features, shape [E, n_f]
            s2 = x[edge_index[1]]  # Target features, shape [E, n_f]

            # Concatenate source and target features as input to msg_fnc
            inp = torch.cat([s1, s2], dim=1)  # Shape [E, 2 * n_f]
            # Compute message vectors
            m12 = model.edge_model(inp)           # Shape [E, msg_dim]

        # Combine source, target, and message arrays, move to CPU and convert to numpy
        combined = torch.cat([s1, s2, m12], dim=1).cpu().numpy()  # [E, 2*n_f + msg_dim]
        all_rows.append(combined)

    # Stack all batches into one array, shape [total_edges, 2*n_f + msg_dim]
    all_arr = np.vstack(all_rows)

    # Generate column names
    cols = []
    # Source node feature names 
    feature_names = []
    # First dim coordinates and velocities
    coord_names = ['x', 'y', 'z'][:dim]
    vel_names = ['vx', 'vy', 'vz'][:dim]
    feature_names.extend(coord_names)
    feature_names.extend(vel_names)
    # Then charge and mass
    feature_names.append('q')
    feature_names.append('m')  # Placeholder for mass or other features
    
    # Build source‐node columns
    for name in feature_names:
        cols.append(f"{name}1")
    # Target node feature names (e.g., x2, y2, vx2, vy2, q2)
    for name in feature_names:
        cols.append(f"{name}2")
    # Message vector component names (msg0, msg1, ...)
    for i in range(msg_dim):
        cols.append(f"msg{i}")

    # Build DataFrame
    df = pd.DataFrame(all_arr, columns=cols)

    # Compute relative displacements and distances
    df['dx'] = df['x1'] - df['x2']
    df['dy'] = df['y1'] - df['y2']
    if dim == 3:
        df['dz'] = df['z1'] - df['z2']
        df['r'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
    else:
        df['r'] = np.sqrt(df['dx']**2 + df['dy']**2)

    return df


def fit_multioutput_force(latent_df, dim):
    """
    Fit a multi-output linear model from true force components to the top-D message channels.

    Parameters:
        latent_df: pd.DataFrame containing columns 'fx_true','fy_true' and 'msg0'...'msgN'
        dim: int, simulation dimensionality (e.g. 2 for 2D, 3 for 3D)

    Returns:
        lr: trained LinearRegression instance
        top_channels: list of selected message channel names
    """
    # Select all message columns
    msg_cols = [c for c in latent_df.columns if c.startswith('msg')]
    # Compute variances and pick top-D channels
    stds = latent_df[msg_cols].std()
    top_channels = stds.nlargest(dim).index.tolist()

    # Prepare design matrix X and multi-output targets Y
    X_force_component = latent_df[['fx_true', 'fy_true']].values          # shape (N, 2)
    Y_message_component = latent_df[top_channels].values                   # shape (N, D)

    # Fit multi-output linear regression
    lr = LinearRegression()
    lr.fit(X_force_component, Y_message_component)

    return lr, top_channels


def measure_variational_latent(model, loader, device, dim, msg_dim):
    """
    Extracts (x1, x2, mu, logvar, dx, dy, r) for every directed edge.
    """
    model.eval()
    rows = []
    for batch in loader:
        batch = batch.to(device)
        x, edge_index = batch.x, batch.edge_index

        with torch.no_grad():
            # 1) gather source/target features
            s_i = x[edge_index[0]]    # [E, n_f]
            s_j = x[edge_index[1]]    # [E, n_f]
            # 2) compute stats = [mu, logvar]
            h     = torch.cat([s_i, s_j], dim=-1)            # [E, 2*n_f]
            stats = model.edge_encoder(h)                    # [E, 2*msg_dim]
            mu, logvar = stats.chunk(2, dim=-1)              # each [E, msg_dim]

        # 3) stack into one array and to CPU
        arr = torch.cat([s_i, s_j, mu, logvar], dim=1).cpu().numpy()  # [E, 2*n_f+2*msg_dim]
        rows.append(arr)

    A = np.vstack(rows)  # [total_edges, 2*n_f + 2*msg_dim]

    # Source node feature names 
    feature_names = []
    # First dim coordinates and velocities
    coord_names = ['x', 'y', 'z'][:dim]
    vel_names = ['vx', 'vy', 'vz'][:dim]
    feature_names.extend(coord_names)
    feature_names.extend(vel_names)
    # Then charge and mass
    feature_names.append('q')
    feature_names.append('m')  # Placeholder for mass or other features
    cols = (
        [f"{name}1" for name in feature_names] +
        [f"{name}2" for name in feature_names] +
        [f"mu{i}"     for i in range(msg_dim)] +
        [f"logvar{i}" for i in range(msg_dim)]
    )
    df = pd.DataFrame(A, columns=cols)

    # compute dx, dy, r
    df['dx'] = df['x1'] - df['x2']
    df['dy'] = df['y1'] - df['y2']
    df['r']  = np.sqrt(df.dx**2 + df.dy**2)
    return df



# Extension: try to fit each channel to each force dimension 
def fit_each_channel_by_force_dim(latent_df, dim):
    """
    For the top-D message channels, perform univariate linear regression using fx_true and fy_true.

    Parameters:
        latent_df (pd.DataFrame): Must contain columns 'fx_true', 'fy_true', and 'msg0', 'msg1', ...
        dim (int): Number of channels to select (typically equal to the spatial dimensionality).

    Returns:
        top_channels (list[str]):
            Names of the selected dim message channels.
        models (dict):
            models[ch]['fx'] = LinearRegression model for fx_true.
            models[ch]['fy'] = LinearRegression model for fy_true.
        r2_scores (dict):
            r2_scores[ch]['fx'] = R² score for the regression using fx_true.
            r2_scores[ch]['fy'] = R² score for the regression using fy_true.
    """
    # Top-D message channels based on variance
    msg_cols = [c for c in latent_df.columns if c.startswith("msg")]
    stds = latent_df[msg_cols].std()
    top_channels = stds.nlargest(dim).index.tolist()

    models = {}
    r2_scores = {}

    # For each top channel, fit a linear regression model
    for ch in top_channels:
        y = latent_df[ch].values            # shape (N_edges,)

        models[ch] = {}
        r2_scores[ch] = {}

        # fx_true regression
        X_fx = latent_df[["fx_true"]].values   # shape (N_edges,1)
        lr_fx = LinearRegression().fit(X_fx, y)
        models[ch]['fx'] = lr_fx
        r2_scores[ch]['fx'] = lr_fx.score(X_fx, y)

        # fy_true regression
        X_fy = latent_df[["fy_true"]].values   # shape (N_edges,1)
        lr_fy = LinearRegression().fit(X_fy, y)
        models[ch]['fy'] = lr_fy
        r2_scores[ch]['fy'] = lr_fy.score(X_fy, y)

    return top_channels, models, r2_scores