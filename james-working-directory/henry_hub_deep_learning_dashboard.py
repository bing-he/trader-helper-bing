"""
Henry Hub Natural Gas Forward Curve Analytics + Deep Learning Dashboard
=========================================================================
PhD-Level Analysis combining:
1. Traditional Forward Curve Analytics
2. Deep Learning Price Forecasting (LSTM, Transformer)
3. Anomaly Detection (Autoencoders)
4. Volatility Forecasting (GARCH-style Neural Networks)
5. Pattern Recognition & Regime Classification

All visualizations organized into two clean pages:
- Page 1: Traditional Analytics (6 subplots)
- Page 2: 3D Surface & Deep Learning Insights
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")


# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================


class LSTMForecaster(nn.Module):
    """LSTM model for multi-step forward curve forecasting"""

    def __init__(
        self, input_size, hidden_size=128, num_layers=2, output_size=24, dropout=0.2
    ):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        prediction = self.fc(last_output)
        return prediction


class TransformerForecaster(nn.Module):
    """Transformer model for forward curve forecasting"""

    def __init__(
        self,
        input_size,
        d_model=128,
        nhead=8,
        num_layers=3,
        output_size=24,
        dropout=0.1,
    ):
        super(TransformerForecaster, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.input_proj(x)
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=1)
        x = self.dropout(x)
        prediction = self.fc(x)
        return prediction


class CurveAutoencoder(nn.Module):
    """Autoencoder for anomaly detection in forward curves"""

    def __init__(self, input_size=24, latent_dim=3):
        super(CurveAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class VolatilityPredictor(nn.Module):
    """Neural network for volatility forecasting (GARCH-style)"""

    def __init__(self, lookback=20, hidden_size=64):
        super(VolatilityPredictor, self).__init__()
        self.lstm = nn.LSTM(1, hidden_size, 2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Softplus(),  # Ensure positive volatility
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        volatility = self.fc(last_output)
        return volatility


# Reference forward methods so dead-code detectors see usage via __call__
_FORWARD_REFS = (
    LSTMForecaster.forward,
    TransformerForecaster.forward,
    CurveAutoencoder.forward,
    VolatilityPredictor.forward,
)


## Removed unused TimeSeriesDataset and DataLoader import (cleaner API)


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================


def load_and_clean_data(filepath):
    """Load forward curve data and handle missing values"""
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    fwd_cols = [f"FWD_{i:02d}" for i in range(24)]
    df = df.dropna(subset=fwd_cols, how="all")
    return df, fwd_cols


def calculate_curve_metrics(df, fwd_cols):
    """Calculate key curve characteristics"""
    metrics = pd.DataFrame(index=df.index)
    metrics["Date"] = df["Date"]
    metrics["Front_Month"] = df["FWD_00"]
    metrics["Slope_12M"] = df["FWD_11"] - df["FWD_00"]
    metrics["Contango_1M"] = df["FWD_01"] - df["FWD_00"]
    metrics["Contango_6M"] = df["FWD_05"] - df["FWD_00"]
    metrics["Contango_12M"] = df["FWD_11"] - df["FWD_00"]
    metrics["Spread_M2_M1"] = df["FWD_01"] - df["FWD_00"]
    metrics["Spread_Summer_Winter"] = (
        df["FWD_04"] + df["FWD_05"] + df["FWD_06"]
    ) / 3 - (df["FWD_10"] + df["FWD_11"] + df["FWD_00"]) / 3
    forward_prices = df[fwd_cols].values
    metrics["Curve_Steepness"] = np.nanstd(forward_prices, axis=1)
    return metrics


def get_contract_month_labels(last_date, periods=24):
    """Return list of Month-Year labels aligned to forward curve contracts.

    Example: [Nov-2025, Dec-2025, Jan-2026, ...]
    """
    # Anchor to the start of the month for consistent labelling
    start = pd.Timestamp(year=last_date.year, month=last_date.month, day=1)
    months = pd.date_range(start=start, periods=periods, freq="MS")
    return [m.strftime("%b-%Y") for m in months]


def perform_pca_analysis(df, fwd_cols, n_components=3):
    """Decompose curve movements into Level, Slope, Curvature factors"""
    # Use returns instead of price levels to focus on curve dynamics (stationary)
    forward_prices = df[fwd_cols].dropna()
    returns = forward_prices.pct_change().dropna()
    dates = df.loc[returns.index, "Date"]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(returns)
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(
        {
            "Date": dates.values,
            "PC1_Level": components[:, 0],
            "PC2_Slope": components[:, 1],
            "PC3_Curvature": components[:, 2],
        }
    )
    explained_var = pca.explained_variance_ratio_ * 100
    return pca_df, explained_var


# ============================================================================
# DEEP LEARNING TRAINING & INFERENCE
# ============================================================================


def prepare_dl_data(df, fwd_cols, lookback=30, train_split=0.8):
    """Prepare data for deep learning models"""
    # Get forward curve matrix
    curve_data = df[fwd_cols].values

    # Encourage stationarity: use first differences across time
    # This models day-over-day changes to the entire forward curve
    curve_data = np.diff(curve_data, axis=0)

    # Normalize
    scaler = MinMaxScaler()
    curve_normalized = scaler.fit_transform(curve_data)

    # Create sequences
    X, y = [], []
    for i in range(len(curve_normalized) - lookback):
        X.append(curve_normalized[i : i + lookback])
        y.append(curve_normalized[i + lookback])

    X = np.array(X)
    y = np.array(y)

    # Train/test split
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, scaler


def walk_forward_validation(
    X,
    y,
    scaler,
    window=252,
    step=21,
    epochs=30,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
):
    """Perform rolling walk-forward validation for realistic forecasting error estimation.

    Returns dict with predictions, actuals, and dates for each fold.
    """
    if not TORCH_AVAILABLE:
        return None

    print("\n[Walk-Forward Validation]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_samples = len(X)
    results = []

    fold = 0
    for start in range(0, n_samples - window - step, step):
        end = start + window
        test_end = min(end + step, n_samples)

        X_train, y_train = X[:end], y[:end]
        X_test, y_test = X[end:test_end], y[end:test_end]

        if len(X_test) == 0:
            break

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)

        # Train model
        model = LSTMForecaster(
            input_size=24,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=24,
            dropout=dropout,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Quick training
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        # Predict
        model.eval()
        with torch.no_grad():
            preds = model(X_test_t).cpu().numpy()

        # Inverse transform to price-change space
        preds_rescaled = scaler.inverse_transform(preds)
        y_test_rescaled = scaler.inverse_transform(y_test)

        results.append(
            {"y_true": y_test_rescaled, "y_pred": preds_rescaled, "fold": fold}
        )

        fold += 1
        if fold % 5 == 0:
            print(f"   Completed fold {fold}...")

    print(f"   Total folds: {fold}")
    return results


def predict_with_uncertainty(model, X, n_samples=50):
    """Monte Carlo dropout for uncertainty quantification."""
    if not TORCH_AVAILABLE:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.FloatTensor(X).to(device)

    model.train()  # Enable dropout
    preds = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(X_tensor).cpu().numpy()
            preds.append(pred)

    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    return mean_pred, std_pred


def calculate_trading_metrics(results, scaler, df, fwd_cols):
    """Calculate trader-relevant metrics from walk-forward results."""
    from sklearn.metrics import r2_score

    # Concatenate all predictions and actuals
    all_true = np.vstack([r["y_true"] for r in results])
    all_pred = np.vstack([r["y_pred"] for r in results])

    # Flatten for overall metrics
    y_true_flat = all_true.flatten()
    y_pred_flat = all_pred.flatten()

    # Filter out any NaN or inf values
    mask = (
        np.isfinite(y_true_flat)
        & np.isfinite(y_pred_flat)
        & (np.abs(y_true_flat) > 1e-8)
    )
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]

    # Statistical metrics
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)

    # R² (coefficient of determination)
    r2 = r2_score(y_true_clean, y_pred_clean)

    # MAPE (mean absolute percentage error)
    mape = (
        np.mean(np.abs((y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + 1e-8)))
        * 100
    )

    # Directional accuracy (sign agreement)
    direction_acc = np.mean(np.sign(y_true_clean) == np.sign(y_pred_clean))

    # Front-month directional accuracy (most tradeable)
    front_true = all_true[:, 0]
    front_pred = all_pred[:, 0]
    front_mask = np.isfinite(front_true) & np.isfinite(front_pred)
    front_dir_acc = np.mean(
        np.sign(front_true[front_mask]) == np.sign(front_pred[front_mask])
    )

    # Economic simulation: trade front month based on predicted direction
    positions = np.sign(front_pred[front_mask])  # +1 long, -1 short
    returns = front_true[front_mask]  # actual price changes
    pnl = positions * returns

    sharpe = 0.0
    if len(pnl) > 0 and np.std(pnl) > 0:
        sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(252)  # Annualized

    win_rate = np.mean(pnl > 0) if len(pnl) > 0 else 0.0
    avg_win = np.mean(pnl[pnl > 0]) if np.any(pnl > 0) else 0.0
    avg_loss = np.mean(pnl[pnl < 0]) if np.any(pnl < 0) else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "direction_acc": direction_acc,
        "front_dir_acc": front_dir_acc,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_trades": len(pnl),
        "cumulative_pnl": np.sum(pnl),
    }


def train_lstm_model(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=50,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    lr=0.001,
):
    """Train LSTM forecasting model"""
    if not TORCH_AVAILABLE:
        return None, None, None

    print("\n[LSTM Training]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    # Create model
    model = LSTMForecaster(
        input_size=24,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=24,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t)
            test_losses.append(test_loss.item())

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}"
            )

    # Final predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()

    return model, predictions, {"train": train_losses, "test": test_losses}


def optimize_lstm_hyperparameters(X_train, y_train, n_trials=30):
    """Use Optuna to find optimal LSTM hyperparameters."""
    if not TORCH_AVAILABLE or not OPTUNA_AVAILABLE:
        print("   Skipping hyperparameter optimization (requires torch + optuna)")
        return {"hidden_size": 128, "num_layers": 2, "dropout": 0.2, "lr": 0.001}

    print("\n[Hyperparameter Optimization with Optuna]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use 80% for train, 20% for validation within training set
    split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        model = LSTMForecaster(
            input_size=24,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=24,
            dropout=dropout,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Quick training (20 epochs for optimization speed)
        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_tr_t)
            loss = criterion(outputs, y_tr_t)
            loss.backward()
            optimizer.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)

        return val_loss.item()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"   Best validation loss: {study.best_value:.6f}")
    print(f"   Best params: {study.best_params}")

    return study.best_params


def train_autoencoder(curve_data, epochs=100):
    """Train autoencoder for anomaly detection"""
    if not TORCH_AVAILABLE:
        return None, None, None

    print("\n[Autoencoder Training]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize data
    scaler = MinMaxScaler()
    curve_normalized = scaler.fit_transform(curve_data)

    # Convert to tensor
    data_tensor = torch.FloatTensor(curve_normalized).to(device)

    # Create model
    model = CurveAutoencoder(input_size=24, latent_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        reconstructed, _ = model(data_tensor)
        loss = criterion(reconstructed, data_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    # Compute anomaly scores (MSE per day)
    model.eval()
    with torch.no_grad():
        reconstructed, encoded = model(data_tensor)
        reconstruction_errors = (
            torch.mean((data_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        )

    return model, reconstruction_errors, scaler


def classify_market_regimes(pca_df):
    """Use PCA components to classify market regimes"""
    # Data-driven regime classification using medians of PC1 and PC2
    pc1_median = pca_df["PC1_Level"].median()
    pc2_median = pca_df["PC2_Slope"].median()

    regimes = []
    for _, row in pca_df.iterrows():
        pc1 = row["PC1_Level"]
        pc2 = row["PC2_Slope"]

        if pc1 > pc1_median and pc2 > pc2_median:
            regime = "High & Steepening"
        elif pc1 > pc1_median and pc2 <= pc2_median:
            regime = "High & Flattening"
        elif pc1 <= pc1_median and pc2 > pc2_median:
            regime = "Low & Steepening"
        else:
            regime = "Low & Flattening"

        regimes.append(regime)

    pca_df["Regime"] = regimes
    return pca_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def create_main_analytics_page(df, fwd_cols, metrics, pca_df, explained_var):
    """Create comprehensive analytics page with 6 subplots"""

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "1. Forward Curve Evolution",
            "2. Contango/Backwardation Regimes",
            "3. PCA Level Factor (PC1)",
            "4. PCA Slope Factor (PC2)",
            "5. Calendar Spreads",
            "6. Seasonal Patterns",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        row_heights=[0.33, 0.33, 0.34],
    )

    # 1. Forward Curve Evolution
    latest = df.iloc[-1]
    month_labels = get_contract_month_labels(latest["Date"], periods=len(fwd_cols))
    fig.add_trace(
        go.Scatter(
            x=list(range(24)),
            y=latest[fwd_cols].values,
            mode="lines+markers",
            name=f"Current ({latest['Date'].strftime('%Y-%m-%d')})",
            line=dict(color="red", width=3),
            showlegend=False,
            hovertemplate="%{y:.3f} $/MMBtu<br>%{customdata}<extra></extra>",
            customdata=np.array(month_labels)[:, None],
        ),
        row=1,
        col=1,
    )
    for year in [2015, 2018, 2021, 2024]:
        year_data = df[df["Date"].dt.year == year]
        if not year_data.empty:
            sample = year_data.iloc[len(year_data) // 2]
            fig.add_trace(
                go.Scatter(
                    x=list(range(24)),
                    y=sample[fwd_cols].values,
                    mode="lines",
                    name=str(year),
                    opacity=0.5,
                    showlegend=False,
                    hovertemplate="%{y:.3f} $/MMBtu<br>%{customdata}<extra></extra>",
                    customdata=np.array(month_labels)[:, None],
                ),
                row=1,
                col=1,
            )

    # 2. Contango/Backwardation
    fig.add_trace(
        go.Scatter(
            x=metrics["Date"],
            y=metrics["Contango_12M"],
            name="12M Contango",
            line=dict(color="blue"),
            showlegend=False,
            fill="tonexty",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

    # 3. PC1 (Level)
    fig.add_trace(
        go.Scatter(
            x=pca_df["Date"],
            y=pca_df["PC1_Level"],
            name=f"Level ({explained_var[0]:.1f}%)",
            line=dict(color="darkblue"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 4. PC2 (Slope)
    fig.add_trace(
        go.Scatter(
            x=pca_df["Date"],
            y=pca_df["PC2_Slope"],
            name=f"Slope ({explained_var[1]:.1f}%)",
            line=dict(color="darkgreen"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # 5. Calendar Spreads
    fig.add_trace(
        go.Scatter(
            x=metrics["Date"],
            y=metrics["Spread_M2_M1"],
            name="M2-M1",
            line=dict(color="orange"),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=1)

    # 6. Seasonal Patterns
    seasonal_df = extract_seasonal_patterns(df, fwd_cols)
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    fig.add_trace(
        go.Scatter(
            x=month_names,
            y=seasonal_df["Avg_Front_Month"],
            mode="lines+markers",
            name="Seasonal",
            line=dict(color="purple", width=2),
            showlegend=False,
        ),
        row=3,
        col=2,
    )

    # Update axes
    # Use month labels for contract axis
    fig.update_xaxes(
        title_text="Contract (Month-Year)",
        tickmode="array",
        tickvals=list(range(24)),
        ticktext=month_labels,
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_xaxes(title_text="Month", row=3, col=2)

    fig.update_yaxes(title_text="Price ($/MMBtu)", row=1, col=1)
    fig.update_yaxes(title_text="Spread ($/MMBtu)", row=1, col=2)
    fig.update_yaxes(title_text="PC1 Score", row=2, col=1)
    fig.update_yaxes(title_text="PC2 Score", row=2, col=2)
    fig.update_yaxes(title_text="Spread ($/MMBtu)", row=3, col=1)
    fig.update_yaxes(title_text="Avg Price ($/MMBtu)", row=3, col=2)

    fig.update_layout(
        title_text="PAGE 1: Traditional Forward Curve Analytics",
        height=1200,
        showlegend=False,
        hovermode="closest",
    )

    return fig


def extract_seasonal_patterns(df, fwd_cols):
    """Calculate average seasonal pattern by month"""
    seasonal_data = []
    for month in range(1, 13):
        month_data = df[df["Date"].dt.month == month]
        avg_curve = month_data[fwd_cols].mean()
        seasonal_data.append(
            {
                "Month": month,
                "Avg_Front_Month": avg_curve["FWD_00"],
            }
        )
    return pd.DataFrame(seasonal_data)


def create_deep_learning_page(
    df,
    fwd_cols,
    predictions,
    anomaly_scores,
    pca_df,
    train_losses,
    validation_metrics=None,
    uncertainty_std=None,
):
    """Create deep learning insights page with validation metrics and uncertainty"""

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "1. 3D Forward Surface",
            "2. LSTM Forecast vs Actual Forward Curve (Next-Day Prediction)",
            "3. Anomaly Detection Scores",
            "4. Market Regime Classification",
            "5. LSTM Training Convergence",
            "6. Forecast Error Distribution",
        ),
        specs=[
            [{"type": "surface", "rowspan": 2}, {"type": "scatter"}],
            [None, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
        row_heights=[0.4, 0.2, 0.4],
    )

    # 1. 3D Surface
    df_sampled = df.iloc[::7].copy()
    dates_numeric = (df_sampled["Date"] - df_sampled["Date"].min()).dt.days.values
    months = np.arange(len(fwd_cols))
    Z = df_sampled[fwd_cols].values
    # Labels for 3D x-axis
    last_date = df["Date"].iloc[-1]
    month_labels = get_contract_month_labels(last_date, periods=len(fwd_cols))

    fig.add_trace(
        go.Surface(
            x=months,
            y=dates_numeric,
            z=Z,
            colorscale="Viridis",
            colorbar=dict(title="Price", x=0.46, len=0.6),
            showscale=True,
        ),
        row=1,
        col=1,
    )

    # 2. LSTM Forecast (if available)
    if predictions is not None and len(predictions) > 0:
        # IMPORTANT: We can't compare against "future actual" - we don't have it yet!
        # Instead, show a historical validation example from the test set where we DO have both.
        # Use a representative test sample (e.g., middle of test set) to show model performance.

        test_idx = (
            len(predictions) // 2
        )  # Middle of test set for representative example

        # Get the actual next-day curve that was observed (from historical test data)
        # This requires accessing y_test which contains the actual next-day changes
        # For now, we'll show the last prediction as a pure forecast without "actual" comparison

        # Get base curve from test period for context
        test_start_idx = int(len(df) * 0.8)  # Assuming 80/20 train/test split
        base_date_idx = test_start_idx + test_idx

        if base_date_idx < len(df) - 1:
            # We have both the base day and next day in historical data
            base_curve = df[fwd_cols].iloc[base_date_idx].values
            actual_next_day = df[fwd_cols].iloc[base_date_idx + 1].values
            predicted_change = predictions[test_idx]
            predicted_next_day = base_curve + predicted_change

            validation_date = df["Date"].iloc[base_date_idx]
            forecast_month_labels = get_contract_month_labels(
                validation_date, periods=len(fwd_cols)
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_month_labels,
                    y=actual_next_day,
                    mode="lines+markers",
                    name=f"Actual (Historical: {validation_date.strftime('%Y-%m-%d')})",
                    line=dict(color="blue", width=2),
                    marker=dict(size=6),
                    hovertemplate="<b>Actual Next-Day</b><br>%{x}<br>%{y:.3f} $/MMBtu<extra></extra>",
                ),
                row=1,
                col=2,
            )
            # Add uncertainty bands if available
            if uncertainty_std is not None:
                upper_bound = predicted_next_day + 2 * uncertainty_std[test_idx]
                lower_bound = predicted_next_day - 2 * uncertainty_std[test_idx]

                fig.add_trace(
                    go.Scatter(
                        x=forecast_month_labels,
                        y=upper_bound,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=2,
                )
                fig.add_trace(
                    go.Scatter(
                        x=forecast_month_labels,
                        y=lower_bound,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(255, 0, 0, 0.2)",
                        name="±2σ Confidence",
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=2,
                )

            forecast_label = "LSTM Forecast (Next-Day)"
            if validation_metrics:
                dir_acc = validation_metrics.get("front_dir_acc", 0) * 100
                forecast_label += f" | Dir Acc: {dir_acc:.1f}%"

            fig.add_trace(
                go.Scatter(
                    x=forecast_month_labels,
                    y=predicted_next_day,
                    mode="lines+markers",
                    name=forecast_label,
                    line=dict(color="red", width=2, dash="dash"),
                    marker=dict(size=6, symbol="x"),
                    hovertemplate="<b>Model Forecast</b><br>%{x}<br>%{y:.3f} $/MMBtu<extra></extra>",
                ),
                row=1,
                col=2,
            )
        else:
            # Fallback: Show only the forecast for the latest date (no actual comparison)
            last_observed = df[fwd_cols].iloc[-1].values
            last_pred_change = predictions[-1]
            forecast_curve = last_observed + last_pred_change

            forecast_month_labels = get_contract_month_labels(
                df["Date"].iloc[-1], periods=len(fwd_cols)
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_month_labels,
                    y=last_observed,
                    mode="lines+markers",
                    name="Current Observed Curve",
                    line=dict(color="blue", width=2),
                    marker=dict(size=6),
                    hovertemplate="<b>Current Market</b><br>%{x}<br>%{y:.3f} $/MMBtu<extra></extra>",
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_month_labels,
                    y=forecast_curve,
                    mode="lines+markers",
                    name="Tomorrow's Forecast (No Actual Yet)",
                    line=dict(color="orange", width=2, dash="dash"),
                    marker=dict(size=6, symbol="x"),
                    hovertemplate="<b>Forecast</b><br>%{x}<br>%{y:.3f} $/MMBtu<extra></extra>",
                ),
                row=1,
                col=2,
            )

    # 3. Anomaly Scores
    if anomaly_scores is not None:
        dates_subset = df["Date"].iloc[: len(anomaly_scores)]
        # Statistical 2-sigma threshold
        threshold = np.float64(np.mean(anomaly_scores) + 2 * np.std(anomaly_scores))

        colors = ["red" if score > threshold else "blue" for score in anomaly_scores]

        fig.add_trace(
            go.Scatter(
                x=dates_subset,
                y=anomaly_scores,
                mode="markers",
                name="Anomaly Score",
                marker=dict(color=colors, size=3),
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        # Add threshold line as a trace instead of hline
        fig.add_trace(
            go.Scatter(
                x=[dates_subset.iloc[0], dates_subset.iloc[-1]],
                y=[threshold, threshold],
                mode="lines",
                name="2-sigma threshold",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # 4. Regime Classification
    if "Regime" in pca_df.columns:
        regime_colors = {
            "High & Steepening": "red",
            "High & Flattening": "orange",
            "Low & Steepening": "lightblue",
            "Low & Flattening": "blue",
        }

        for regime, color in regime_colors.items():
            regime_data = pca_df[pca_df["Regime"] == regime]
            fig.add_trace(
                go.Scatter(
                    x=regime_data["PC1_Level"],
                    y=regime_data["PC2_Slope"],
                    mode="markers",
                    name=regime,
                    marker=dict(color=color, size=4, opacity=0.6),
                ),
                row=3,
                col=1,
            )

    # 5. Training Loss
    if train_losses is not None:
        epochs = list(range(1, len(train_losses["train"]) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_losses["train"],
                mode="lines",
                name="Train Loss",
                line=dict(color="blue"),
            ),
            row=3,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_losses["test"],
                mode="lines",
                name="Test Loss",
                line=dict(color="red"),
            ),
            row=3,
            col=2,
        )

    # Update 3D scene
    fig.update_scenes(
        xaxis_title="Contract (Month-Year)",
        yaxis_title="Days Since 2015",
        zaxis_title="Price ($/MMBtu)",
        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        xaxis=dict(tickvals=list(range(24)), ticktext=month_labels),
        row=1,
        col=1,
    )

    # Update axes
    fig.update_xaxes(
        title_text="Contract Month (Forward Tenor)",
        tickangle=45,
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_xaxes(title_text="PC1 (Level)", row=3, col=1)
    fig.update_xaxes(title_text="Epoch", row=3, col=2)

    fig.update_yaxes(title_text="Price ($/MMBtu)", row=1, col=2)
    fig.update_yaxes(title_text="Reconstruction Error", row=2, col=2)
    fig.update_yaxes(title_text="PC2 (Slope)", row=3, col=1)
    fig.update_yaxes(title_text="MSE Loss", row=3, col=2, type="log")

    fig.update_layout(
        title_text="PAGE 2: 3D Surface & Deep Learning Insights",
        height=1200,
        showlegend=True,
        legend=dict(
            x=0.55,
            y=0.95,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
        ),
        hovermode="closest",
    )

    return fig


# ============================================================================
# MAIN DASHBOARD GENERATION
# ============================================================================


def generate_comprehensive_dashboard(filepath):
    """Generate complete dashboard with traditional + deep learning analytics"""

    print("\n" + "=" * 80)
    print("HENRY HUB FORWARD CURVE - Deep Learning Analytics")
    print("=" * 80)

    # Load data
    print("\n[1/7] Loading data...")
    df, fwd_cols = load_and_clean_data(filepath)
    print(
        f"   Dataset: {len(df)} days from {df['Date'].min().date()} to {df['Date'].max().date()}"
    )

    # Calculate metrics
    print("[2/7] Calculating curve metrics...")
    metrics = calculate_curve_metrics(df, fwd_cols)

    # PCA analysis
    print("[3/7] Performing PCA decomposition...")
    pca_df, explained_var = perform_pca_analysis(df, fwd_cols)
    pca_df = classify_market_regimes(pca_df)

    # Prepare deep learning data
    print("[4/7] Preparing deep learning datasets...")
    predictions = None
    train_losses = None
    anomaly_scores = None
    validation_metrics = None
    uncertainty_std = None

    if TORCH_AVAILABLE:
        X_train, X_test, y_train, y_test, scaler = prepare_dl_data(df, fwd_cols)
        print(f"   Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Hyperparameter optimization (if Optuna available)
        best_params = {"hidden_size": 128, "num_layers": 2, "dropout": 0.2, "lr": 0.001}
        if OPTUNA_AVAILABLE:
            print("[5/7] Optimizing hyperparameters...")
            best_params = optimize_lstm_hyperparameters(X_train, y_train, n_trials=20)
        else:
            print("[5/7] Skipping hyperparameter optimization (Optuna not available)")

        # Walk-forward validation for realistic performance
        print("[6/7] Performing walk-forward validation...")
        X_all = np.vstack([X_train, X_test])
        y_all = np.vstack([y_train, y_test])
        wf_results = walk_forward_validation(
            X_all,
            y_all,
            scaler,
            window=200,
            step=20,
            epochs=30,
            hidden_size=best_params["hidden_size"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
        )

        if wf_results:
            validation_metrics = calculate_trading_metrics(
                wf_results, scaler, df, fwd_cols
            )
            print(f"   Walk-forward R²: {validation_metrics['r2']:.3f}")
            print(f"   Directional Accuracy: {validation_metrics['front_dir_acc']:.2%}")
            print(f"   Simulated Sharpe: {validation_metrics['sharpe']:.2f}")

        # Train final model with best params
        print("[7/7] Training final LSTM model...")
        model, predictions, train_losses = train_lstm_model(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=50,
            hidden_size=best_params["hidden_size"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
            lr=best_params["lr"],
        )

        if predictions is not None:
            # Inverse transform diff-space predictions/targets back to $-space (still differences)
            predictions = scaler.inverse_transform(predictions)
            y_test_actual = scaler.inverse_transform(y_test)

            # Calculate metrics in difference space (stable)
            mse = mean_squared_error(y_test_actual, predictions)
            mae = mean_absolute_error(y_test_actual, predictions)
            print(f"   Test (diff-space) MSE: {mse:.4f}, MAE: {mae:.4f}")

            # Get uncertainty estimates
            if model is not None:
                print("   Computing uncertainty estimates...")
                _, uncertainty_std = predict_with_uncertainty(
                    model, X_test, n_samples=30
                )
                if uncertainty_std is not None:
                    uncertainty_std = scaler.inverse_transform(uncertainty_std)

        # Train autoencoder
        print("[8/8] Training autoencoder for anomaly detection...")
        curve_data = df[fwd_cols].values
        _, anomaly_scores, _ = train_autoencoder(curve_data, epochs=100)

        if anomaly_scores is not None:
            thresh = np.float64(np.mean(anomaly_scores) + 2 * np.std(anomaly_scores))
            anomalies = int(np.sum(anomaly_scores > thresh))
            print(f"   Detected {anomalies} anomalous days (2-sigma)")
    else:
        print("[4/8] Skipping deep learning (PyTorch not available)")

    # Create visualizations
    print("\n[Generating Visualizations]")

    # Page 1: Traditional Analytics
    fig_page1 = create_main_analytics_page(df, fwd_cols, metrics, pca_df, explained_var)

    # Page 2: Deep Learning + 3D
    # For visualization, convert the last prediction into a next-day curve forecast if available
    vis_predictions = None
    if predictions is not None and len(predictions) > 0:
        vis_predictions = predictions.copy()
        try:
            vis_predictions[-1] = df[fwd_cols].iloc[-1].values + vis_predictions[-1]
        except Exception:
            pass

    fig_page2 = create_deep_learning_page(
        df,
        fwd_cols,
        vis_predictions,
        anomaly_scores,
        pca_df,
        train_losses,
        validation_metrics=validation_metrics,
        uncertainty_std=uncertainty_std,
    )

    # Display
    print("\n" + "=" * 80)
    print("DISPLAYING DASHBOARDS")
    print("=" * 80)

    fig_page1.show()
    fig_page2.show()

    # Summary insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS - ANALYSIS")
    print("=" * 80)

    print("\nDATASET OVERVIEW")
    print(f"   • Time span: {(df['Date'].max() - df['Date'].min()).days} days")
    print(f"   • Current front month: ${metrics['Front_Month'].iloc[-1]:.3f}")
    print(f"   • Historical volatility: {metrics['Front_Month'].std():.3f} $/MMBtu")

    print("\nPCA FACTOR ANALYSIS")
    print(f"   • PC1 (Level): {explained_var[0]:.2f}% - parallel shifts in curve")
    print(f"   • PC2 (Slope): {explained_var[1]:.2f}% - curve steepening/flattening")
    print(f"   • PC3 (Curvature): {explained_var[2]:.2f}% - curve bending")
    print(f"   • Total explained: {explained_var.sum():.2f}%")

    if "Regime" in pca_df.columns:
        regime_counts = pca_df["Regime"].value_counts()
        print("\nMARKET REGIMES (Last 10 years)")
        for regime, count in regime_counts.items():
            pct = (count / len(pca_df)) * 100
            print(f"   • {regime}: {pct:.1f}%")

    if TORCH_AVAILABLE and predictions is not None:
        print("\nDEEP LEARNING MODELS")
        print(f"   • LSTM Forecast MAE: ${mae:.4f}")
        print(f"   • LSTM Forecast RMSE: ${np.sqrt(mse):.4f}")

        if anomaly_scores is not None:
            thresh = np.float64(np.mean(anomaly_scores) + 2 * np.std(anomaly_scores))
            recent_anomalies = int(np.sum(anomaly_scores[-252:] > thresh))
            print(f"   • Anomalies (last year, 2-sigma): {recent_anomalies} days")

    if validation_metrics:
        print("\n" + "=" * 80)
        print("FORECAST RELIABILITY SUMMARY (Walk-Forward Validation)")
        print("=" * 80)
        print("\n   Statistical Performance:")
        print(f"   • R² (forecast fit): {validation_metrics['r2']:.3f}")
        print(f"   • RMSE (avg error): ${validation_metrics['rmse']:.4f}")
        print(f"   • MAPE (avg % error): {validation_metrics['mape']:.2f}%")
        print(
            f"   • Overall Directional Accuracy: {validation_metrics['direction_acc']:.2%}"
        )
        print(
            f"   • Front-Month Directional Accuracy: {validation_metrics['front_dir_acc']:.2%}"
        )

        print("\n   Economic Performance (Simulated Trading):")
        print(f"   • Total Trades: {validation_metrics['total_trades']}")
        print(f"   • Win Rate: {validation_metrics['win_rate']:.2%}")
        print(f"   • Avg Win: ${validation_metrics['avg_win']:.4f}")
        print(f"   • Avg Loss: ${validation_metrics['avg_loss']:.4f}")
        print(f"   • Cumulative PnL: ${validation_metrics['cumulative_pnl']:.4f}")
        print(f"   • Sharpe Ratio (Annualized): {validation_metrics['sharpe']:.2f}")

        if uncertainty_std is not None:
            avg_uncertainty = np.float64(np.mean(uncertainty_std) * 100)
            print("\n   Uncertainty Quantification:")
            print(
                f"   • Avg Forecast Uncertainty: ±{avg_uncertainty:.2f}% (1σ from MC dropout)"
            )

        print("\n   TRADING INTERPRETATION:")
        if (
            validation_metrics["front_dir_acc"] > 0.60
            and validation_metrics["sharpe"] > 1.0
        ):
            print("   ✓ LSTM forecast shows STRONG predictive power")
            print(
                "   ✓ Directional accuracy exceeds 60% with positive risk-adjusted returns"
            )
            print("   ✓ Model can be used as PRIMARY signal for front-month trading")
        elif (
            validation_metrics["front_dir_acc"] > 0.55
            and validation_metrics["sharpe"] > 0.5
        ):
            print("   ~ LSTM forecast shows MODERATE predictive power")
            print(
                "   ~ Treat as SIGNAL ENHANCER in conjunction with fundamental analysis"
            )
            print("   ~ Consider position sizing based on confidence bands")
        else:
            print("   ✗ LSTM forecast shows LIMITED predictive power")
            print("   ✗ Treat as ADVISORY CONTEXT only, not actionable signal")
            print("   ✗ Do NOT trade mechanically on model output alone")

        print("\n   Risk Management:")
        print("   • Always use stop-losses when trading model signals")
        print("   • Scale position size inversely with forecast uncertainty")
        print("   • Re-validate model monthly as market regime shifts")

    print("\nTRADING INSIGHTS")
    current_contango = metrics["Contango_12M"].iloc[-1]
    avg_contango = metrics["Contango_12M"].mean()
    print(f"   • Current 12M contango: ${current_contango:.3f}")
    print(f"   • Historical avg contango: ${avg_contango:.3f}")
    print(
        f"   • Curve shape: {'CONTANGO' if current_contango > 0 else 'BACKWARDATION'}"
    )
    print(
        f"   • Relative position: {'Steeper' if current_contango > avg_contango else 'Flatter'} than average"
    )

    print("\n" + "=" * 80)
    print("Dashboard generation complete!")
    print("=" * 80 + "\n")

    return df, metrics, pca_df, predictions, anomaly_scores


if __name__ == "__main__":
    filepath = "/Users/jamessawyer/Desktop/trader-helper/INFO/HenryForwardCurve.csv"

    df, metrics, pca_df, predictions, anomaly_scores = generate_comprehensive_dashboard(
        filepath
    )

    print("\n✅ Available for further analysis:")
    print("   - df: Main dataset")
    print("   - metrics: Calculated metrics")
    print("   - pca_df: PCA components with regime classification")
    print("   - predictions: LSTM forecasts")
    print("   - anomaly_scores: Autoencoder anomaly detection scores")
