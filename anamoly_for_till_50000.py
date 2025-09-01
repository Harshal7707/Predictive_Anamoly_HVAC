import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import altair as alt

st.set_page_config(page_title="HVAC Anomaly Detection (50k)", layout="wide")

@st.cache_data
def load_data(file) -> pd.DataFrame:
    # load only necessary columns
    return pd.read_csv(file, usecols=[
        "Timestamp","T_Supply","T_Return","SP_Return","T_Saturation",
        "T_Outdoor","RH_Supply","RH_Return","RH_Outdoor","Energy","Power"
    ], parse_dates=["Timestamp"])

@st.cache_data
def compute_anomalies(t_supply: np.ndarray):
    # 1. Z-Score
    z = np.abs(stats.zscore(t_supply))
    z_idx = np.where(z > 2.5)[0]

    # 2. Modified Z-Score
    med = np.median(t_supply)
    mad = np.median(np.abs(t_supply - med))
    mz = 0.6745 * np.abs(t_supply - med) / mad
    mz_idx = np.where(mz > 3.5)[0]

    # 3. IQR
    Q1, Q3 = np.percentile(t_supply, [25,75])
    iqr = Q3 - Q1
    iqr_idx = np.where((t_supply < Q1 - 1.5*iqr) | (t_supply > Q3 + 1.5*iqr))[0]

    # 4. Isolation Forest (subsample for speed)
    iso = IsolationForest(n_estimators=50, max_samples=0.1, contamination=0.01, random_state=42)
    labels = iso.fit_predict(t_supply.reshape(-1,1))
    iso_idx = np.where(labels == -1)[0]

    # 5. Percentile
    low, high = np.percentile(t_supply, [5,95])
    pct_idx = np.where((t_supply < low) | (t_supply > high))[0]

    # consensus
    all_sets = [set(z_idx), set(mz_idx), set(iqr_idx), set(iso_idx), set(pct_idx)]
    consensus = sorted(set.intersection(*all_sets))

    return {
        "Z-Score": z_idx.tolist(),
        "Mod Z-Score": mz_idx.tolist(),
        "IQR": iqr_idx.tolist(),
        "IsolationForest": iso_idx.tolist(),
        "Percentile": pct_idx.tolist(),
        "Consensus": consensus
    }

def downsample_df(df: pd.DataFrame, factor: int = 10):
    return df.iloc[::factor, :]

# UI
st.title("HVAC Anomaly Detection (Scalable to 50k rows)")
uploaded = st.file_uploader("Upload HVAC CSV", type="csv")
if not uploaded:
    st.stop()

df = load_data(uploaded)
st.write("Data loaded:", df.shape)

t_supply = df["T_Supply"].values
anoms = compute_anomalies(t_supply)

# Show anomaly summary
st.subheader("Anomaly Summary")
for name, idx in anoms.items():
    st.write(f"{name}: {len(idx)} points")

# Plot downsampled time series with consensus anomalies
factor = max(1, len(df)//5000)
df_plot = downsample_df(df, factor)
src = pd.DataFrame({
    "Timestamp": df_plot["Timestamp"],
    "Value": df_plot["T_Supply"],
    "Anomaly": df_plot.index.isin(anoms["Consensus"])
})
chart = alt.Chart(src).mark_line().encode(
    x="Timestamp:T", y="Value:Q"
) + alt.Chart(src[src["Anomaly"]]).mark_point(color="red", size=50).encode(
    x="Timestamp:T", y="Value:Q"
)
st.altair_chart(chart, use_container_width=True)

# Display first 1000 rows with flag
st.subheader("First 1 000 Rows with Consensus Flag")
df["Anomaly"] = False
df.loc[anoms["Consensus"], "Anomaly"] = True
st.dataframe(df.head(1000))

# Download processed CSV
csv = df.to_csv(index=False).encode()
st.download_button("Download Full Processed CSV", csv, "hvac_50k_processed.csv", "text/csv")
