"""
Henry Hub Natural Gas Forward Curve Analytics Dashboard
========================================================
Comprehensive analysis of NYMEX Henry Hub forward curve time series (2015-2025)

This dashboard provides:
1. Curve Shape Analysis (Contango/Backwardation)
2. Term Structure Evolution
3. Volatility Term Structure
4. PCA Decomposition (Level, Slope, Curvature)
5. Calendar Spreads & Arbitrage Signals
6. Seasonal Patterns
7. 3D Forward Surface Visualization
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================


def load_and_clean_data(filepath):
    """Load forward curve data and handle missing values"""
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])

    # Extract forward price columns
    fwd_cols = [f"FWD_{i:02d}" for i in range(24)]

    # Remove rows with all NaN forwards
    df = df.dropna(subset=fwd_cols, how="all")

    return df, fwd_cols


def calculate_curve_metrics(df, fwd_cols):
    """Calculate key curve characteristics"""
    metrics = pd.DataFrame(index=df.index)
    metrics["Date"] = df["Date"]

    # Front month price
    metrics["Front_Month"] = df["FWD_00"]

    # Curve slope (12M - Front)
    metrics["Slope_12M"] = df["FWD_11"] - df["FWD_00"]

    # Contango/Backwardation indicator
    metrics["Contango_1M"] = df["FWD_01"] - df["FWD_00"]  # M2-M1
    metrics["Contango_6M"] = df["FWD_05"] - df["FWD_00"]  # M6-M1
    metrics["Contango_12M"] = df["FWD_11"] - df["FWD_00"]  # M12-M1

    # Calendar spreads (key trading pairs)
    metrics["Spread_M2_M1"] = df["FWD_01"] - df["FWD_00"]
    metrics["Spread_M3_M1"] = df["FWD_02"] - df["FWD_00"]
    metrics["Spread_Summer_Winter"] = (
        (df["FWD_04"] + df["FWD_05"] + df["FWD_06"]) / 3  # Summer
        - (df["FWD_10"] + df["FWD_11"] + df["FWD_00"]) / 3  # Winter
    )

    # Curve steepness (std dev of entire curve)
    forward_prices = df[fwd_cols].values
    metrics["Curve_Steepness"] = np.nanstd(forward_prices, axis=1)

    # Maximum price along curve
    metrics["Curve_Peak_Price"] = np.nanmax(forward_prices, axis=1)
    metrics["Curve_Peak_Month"] = np.nanargmax(forward_prices, axis=1)

    return metrics


def calculate_volatility_term_structure(df, fwd_cols, window=30):
    """Calculate rolling volatility for each tenor"""
    vol_data = pd.DataFrame(index=df["Date"])

    for col in fwd_cols:
        returns = df[col].pct_change()
        vol_data[col] = (
            returns.rolling(window=window).std() * np.sqrt(252) * 100
        )  # Annualized %

    return vol_data


def perform_pca_analysis(df, fwd_cols, n_components=3):
    """Decompose curve movements into Level, Slope, Curvature factors"""
    # Get forward prices matrix
    forward_prices = df[fwd_cols].dropna()
    dates = df.loc[forward_prices.index, "Date"]

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(forward_prices)

    # Perform PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled_data)

    # Create dataframe with components
    pca_df = pd.DataFrame(
        {
            "Date": dates.values,
            "PC1_Level": components[:, 0],
            "PC2_Slope": components[:, 1],
            "PC3_Curvature": components[:, 2],
        }
    )

    # Explained variance
    explained_var = pca.explained_variance_ratio_ * 100

    # Factor loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1_Level", "PC2_Slope", "PC3_Curvature"],
        index=fwd_cols,
    )

    return pca_df, explained_var, loadings


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
                "Avg_M6": avg_curve["FWD_05"],
                "Avg_M12": avg_curve["FWD_11"],
            }
        )

    return pd.DataFrame(seasonal_data)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def create_3d_forward_surface(df, fwd_cols):
    """Create 3D surface plot of forward curve evolution"""
    # Sample data for performance (every 7 days)
    df_sampled = df.iloc[::7].copy()

    # Prepare data
    dates_numeric = (df_sampled["Date"] - df_sampled["Date"].min()).dt.days.values
    months = np.arange(len(fwd_cols))

    Z = df_sampled[fwd_cols].values

    fig = go.Figure(
        data=[
            go.Surface(
                x=months,
                y=dates_numeric,
                z=Z,
                colorscale="Viridis",
                colorbar=dict(title="Price ($/MMBtu)"),
            )
        ]
    )

    fig.update_layout(
        title="3D Forward Surface (Henry Hub Natural Gas)",
        scene=dict(
            xaxis_title="Contract Month",
            yaxis_title="Days Since 2015",
            zaxis_title="Price ($/MMBtu)",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        height=600,
    )

    return fig


def create_curve_evolution_plot(df, fwd_cols):
    """Plot current vs historical curves"""
    fig = go.Figure()

    # Recent curve
    latest = df.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=list(range(24)),
            y=latest[fwd_cols].values,
            mode="lines+markers",
            name=f"Current ({latest['Date'].strftime('%Y-%m-%d')})",
            line=dict(color="red", width=3),
        )
    )

    # Historical snapshots (yearly)
    for year in [2015, 2017, 2019, 2021, 2023, 2024]:
        year_data = df[df["Date"].dt.year == year]
        if not year_data.empty:
            sample = year_data.iloc[len(year_data) // 2]  # Mid-year
            fig.add_trace(
                go.Scatter(
                    x=list(range(24)),
                    y=sample[fwd_cols].values,
                    mode="lines",
                    name=str(year),
                    opacity=0.6,
                )
            )

    fig.update_layout(
        title="Forward Curve Evolution Over Time",
        xaxis_title="Contract Month (0=Front)",
        yaxis_title="Price ($/MMBtu)",
        hovermode="x unified",
        height=500,
    )

    return fig


def create_contango_backwardation_analysis(metrics):
    """Analyze contango/backwardation regimes"""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Front Month Price", "Contango (+) / Backwardation (-)"),
        vertical_spacing=0.12,
        row_heights=[0.4, 0.6],
    )

    # Front month price
    fig.add_trace(
        go.Scatter(
            x=metrics["Date"],
            y=metrics["Front_Month"],
            name="Front Month",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Contango metrics
    fig.add_trace(
        go.Scatter(
            x=metrics["Date"],
            y=metrics["Contango_1M"],
            name="1M Spread",
            line=dict(color="lightblue"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=metrics["Date"],
            y=metrics["Contango_6M"],
            name="6M Spread",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=metrics["Date"],
            y=metrics["Contango_12M"],
            name="12M Spread",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )

    # Zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($/MMBtu)", row=1, col=1)
    fig.update_yaxes(title_text="Spread ($/MMBtu)", row=2, col=1)

    fig.update_layout(height=700, showlegend=True, hovermode="x unified")

    return fig


def create_volatility_term_structure_plot(vol_data, df):
    """Plot volatility by tenor over time"""
    # Create heatmap
    vol_matrix = vol_data.T

    fig = go.Figure(
        data=go.Heatmap(
            z=vol_matrix.values,
            x=vol_data.index,
            y=list(range(24)),
            colorscale="YlOrRd",
            colorbar=dict(title="Volatility (%)"),
            hovertemplate="Date: %{x}<br>Month: %{y}<br>Vol: %{z:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Volatility Term Structure Heatmap (30-day rolling)",
        xaxis_title="Date",
        yaxis_title="Contract Month",
        height=500,
    )

    return fig


def create_pca_analysis_plot(pca_df, explained_var, loadings):
    """Visualize PCA decomposition"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"PC1 - Level ({explained_var[0]:.1f}% var)",
            f"PC2 - Slope ({explained_var[1]:.1f}% var)",
            f"PC3 - Curvature ({explained_var[2]:.1f}% var)",
            "Factor Loadings",
        ),
        specs=[[{}, {}], [{}, {}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # PC1 time series
    fig.add_trace(
        go.Scatter(
            x=pca_df["Date"], y=pca_df["PC1_Level"], name="PC1", line=dict(color="blue")
        ),
        row=1,
        col=1,
    )

    # PC2 time series
    fig.add_trace(
        go.Scatter(
            x=pca_df["Date"],
            y=pca_df["PC2_Slope"],
            name="PC2",
            line=dict(color="green"),
        ),
        row=1,
        col=2,
    )

    # PC3 time series
    fig.add_trace(
        go.Scatter(
            x=pca_df["Date"],
            y=pca_df["PC3_Curvature"],
            name="PC3",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )

    # Factor loadings
    months = list(range(24))
    fig.add_trace(
        go.Scatter(
            x=months, y=loadings["PC1_Level"], name="Level", mode="lines+markers"
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=months, y=loadings["PC2_Slope"], name="Slope", mode="lines+markers"
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=loadings["PC3_Curvature"],
            name="Curvature",
            mode="lines+markers",
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Contract Month", row=2, col=2)
    fig.update_yaxes(title_text="Loading", row=2, col=2)

    fig.update_layout(height=700, showlegend=True)

    return fig


def create_seasonal_analysis_plot(seasonal_df):
    """Plot seasonal patterns"""
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

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=month_names,
            y=seasonal_df["Avg_Front_Month"],
            mode="lines+markers",
            name="Front Month",
            line=dict(color="blue", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=month_names,
            y=seasonal_df["Avg_M6"],
            mode="lines+markers",
            name="6M Forward",
            line=dict(color="orange", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=month_names,
            y=seasonal_df["Avg_M12"],
            mode="lines+markers",
            name="12M Forward",
            line=dict(color="red", width=2),
        )
    )

    fig.update_layout(
        title="Seasonal Price Patterns (2015-2025 Average)",
        xaxis_title="Month",
        yaxis_title="Average Price ($/MMBtu)",
        height=500,
        hovermode="x unified",
    )

    return fig


def create_spread_trading_signals(metrics):
    """Identify spread trading opportunities"""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Calendar Spreads", "Summer-Winter Spread"),
        vertical_spacing=0.15,
    )

    # Calendar spreads
    fig.add_trace(
        go.Scatter(
            x=metrics["Date"],
            y=metrics["Spread_M2_M1"],
            name="M2-M1",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=metrics["Date"],
            y=metrics["Spread_M3_M1"],
            name="M3-M1",
            line=dict(color="orange"),
        ),
        row=1,
        col=1,
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

    # Summer-Winter spread
    fig.add_trace(
        go.Scatter(
            x=metrics["Date"],
            y=metrics["Spread_Summer_Winter"],
            name="Summer-Winter",
            line=dict(color="green"),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Spread ($/MMBtu)", row=1, col=1)
    fig.update_yaxes(title_text="Spread ($/MMBtu)", row=2, col=1)

    fig.update_layout(height=700, showlegend=True, hovermode="x unified")

    return fig


def create_summary_statistics_table(df, fwd_cols, metrics):
    """Generate summary statistics"""
    stats = {"Metric": [], "Value": []}

    # Dataset info
    stats["Metric"].extend(
        [
            "Data Range",
            "Total Days",
            "Front Month - Current",
            "Front Month - Min (All Time)",
            "Front Month - Max (All Time)",
            "Front Month - Avg (All Time)",
            "Current Curve Slope (12M-Front)",
            "Avg Contango/Backwardation",
            "Days in Contango (1M)",
            "Days in Backwardation (1M)",
        ]
    )

    stats["Value"].extend(
        [
            f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
            f"{len(df):,}",
            f"${metrics['Front_Month'].iloc[-1]:.3f}",
            f"${metrics['Front_Month'].min():.3f}",
            f"${metrics['Front_Month'].max():.3f}",
            f"${metrics['Front_Month'].mean():.3f}",
            f"${metrics['Slope_12M'].iloc[-1]:.3f}",
            f"${metrics['Contango_12M'].mean():.3f}",
            f"{(metrics['Contango_1M'] > 0).sum():,} ({(metrics['Contango_1M'] > 0).sum() / len(metrics) * 100:.1f}%)",
            f"{(metrics['Contango_1M'] < 0).sum():,} ({(metrics['Contango_1M'] < 0).sum() / len(metrics) * 100:.1f}%)",
        ]
    )

    stats_df = pd.DataFrame(stats)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>"],
                    fill_color="darkblue",
                    font=dict(color="white", size=14),
                    align="left",
                ),
                cells=dict(
                    values=[stats_df["Metric"], stats_df["Value"]],
                    fill_color="lavender",
                    align="left",
                    font=dict(size=12),
                    height=30,
                ),
            )
        ]
    )

    fig.update_layout(title="Henry Hub Forward Curve - Summary Statistics", height=450)

    return fig


# ============================================================================
# MAIN DASHBOARD GENERATION
# ============================================================================


def generate_dashboard(filepath):
    """Generate comprehensive forward curve dashboard"""

    print("Loading data...")
    df, fwd_cols = load_and_clean_data(filepath)

    print(f"Dataset: {len(df)} days from {df['Date'].min()} to {df['Date'].max()}")
    print(f"Forward contracts: {len(fwd_cols)} months\n")

    print("Calculating metrics...")
    metrics = calculate_curve_metrics(df, fwd_cols)

    print("Calculating volatility term structure...")
    vol_data = calculate_volatility_term_structure(df, fwd_cols)

    print("Performing PCA analysis...")
    pca_df, explained_var, loadings = perform_pca_analysis(df, fwd_cols)

    print("Extracting seasonal patterns...")
    seasonal_df = extract_seasonal_patterns(df, fwd_cols)

    print("\nGenerating visualizations...\n")

    # Create all plots
    figures = {
        "summary": create_summary_statistics_table(df, fwd_cols, metrics),
        "curve_evolution": create_curve_evolution_plot(df, fwd_cols),
        "contango": create_contango_backwardation_analysis(metrics),
        "volatility": create_volatility_term_structure_plot(vol_data, df),
        "pca": create_pca_analysis_plot(pca_df, explained_var, loadings),
        "seasonal": create_seasonal_analysis_plot(seasonal_df),
        "spreads": create_spread_trading_signals(metrics),
        "surface_3d": create_3d_forward_surface(df, fwd_cols),
    }

    # Display all figures
    print("=" * 80)
    print("HENRY HUB NATURAL GAS FORWARD CURVE DASHBOARD")
    print("=" * 80)
    print()

    for name, fig in figures.items():
        print(f"Displaying: {name}")
        fig.show()

    # Save key metrics
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print(f"\n1. CURRENT MARKET STATE ({df['Date'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"   Front Month: ${metrics['Front_Month'].iloc[-1]:.3f}")
    print(f"   12M Forward: ${df['FWD_11'].iloc[-1]:.3f}")
    print(
        f"   Curve Shape: {'CONTANGO' if metrics['Contango_12M'].iloc[-1] > 0 else 'BACKWARDATION'}"
    )
    print(f"   12M Slope: ${metrics['Slope_12M'].iloc[-1]:.3f}")

    print("\n2. HISTORICAL PERSPECTIVE (2015-2025)")
    print(
        f"   All-Time Range: ${metrics['Front_Month'].min():.3f} - ${metrics['Front_Month'].max():.3f}"
    )
    print(f"   Average Price: ${metrics['Front_Month'].mean():.3f}")
    print(
        f"   Time in Contango: {(metrics['Contango_1M'] > 0).sum() / len(metrics) * 100:.1f}%"
    )
    print(f"   Volatility (Front): {metrics['Front_Month'].std():.3f} $/MMBtu")

    print("\n3. PCA DECOMPOSITION")
    print(f"   Level (PC1): {explained_var[0]:.1f}% of variance")
    print(f"   Slope (PC2): {explained_var[1]:.1f}% of variance")
    print(f"   Curvature (PC3): {explained_var[2]:.1f}% of variance")
    print(f"   Total Explained: {explained_var.sum():.1f}%")

    print("\n4. SEASONAL PATTERNS")
    peak_month = seasonal_df.loc[seasonal_df["Avg_Front_Month"].idxmax(), "Month"]
    trough_month = seasonal_df.loc[seasonal_df["Avg_Front_Month"].idxmin(), "Month"]
    print(
        f"   Peak Month: {['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(peak_month)]}"
    )
    print(
        f"   Trough Month: {['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(trough_month)]}"
    )

    print("\n" + "=" * 80)
    print("Dashboard generation complete!")
    print("=" * 80)

    return df, metrics, vol_data, pca_df, seasonal_df, figures


if __name__ == "__main__":
    # Path to the forward curve data
    filepath = "/Users/jamessawyer/Desktop/trader-helper/INFO/HenryForwardCurve.csv"

    # Generate dashboard
    df, metrics, vol_data, pca_df, seasonal_df, figures = generate_dashboard(filepath)

    print("\nData objects available:")
    print("  - df: Main dataset")
    print("  - metrics: Calculated metrics")
    print("  - vol_data: Volatility term structure")
    print("  - pca_df: PCA components")
    print("  - seasonal_df: Seasonal patterns")
    print("  - figures: Dictionary of all plots")
