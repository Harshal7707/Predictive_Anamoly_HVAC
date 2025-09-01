# streamlit_hvac_interactive_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="HVAC Anomaly Detection & Visualization", layout="wide")
st.title("Interactive HVAC Anomaly Detection and Visualization")

st.markdown("""
Upload your HVAC CSV dataset.  
Use the dropdowns to select which column to plot over time, and inspect anomalies in T_Supply.
""")

# File uploader
uploaded_file = st.file_uploader("Upload HVAC CSV file", type="csv")
if uploaded_file is None:
    st.stop()

# Load data
df = pd.read_csv(uploaded_file, parse_dates=["Timestamp"])
st.write("### Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Ensure required columns
required = ["Timestamp","T_Supply","T_Return","SP_Return","T_Saturation",
            "T_Outdoor","RH_Supply","RH_Return","RH_Outdoor","Energy","Power"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# Anomaly detection on T_Supply
data = df["T_Supply"].values
z_scores = np.abs(stats.zscore(data))
mz = 0.6745 * np.abs(data - np.median(data)) / np.median(np.abs(data - np.median(data)))
Q1, Q3 = np.percentile(data, [25,75])
IQR = Q3 - Q1
iso = IsolationForest(contamination=0.1, random_state=42).fit_predict(data.reshape(-1,1))
low_p, high_p = np.percentile(data, [5,95])

methods = {
    "Z-Score (2.5)": np.where(z_scores>2.5)[0],
    "Modified Z-Score (3.5)": np.where(mz>3.5)[0],
    "IQR": np.where((data<Q1-1.5*IQR)|(data>Q3+1.5*IQR))[0],
    "Isolation Forest": np.where(iso==-1)[0],
    "Percentile (5–95)": np.where((data<low_p)|(data>high_p))[0]
}
consensus = set.intersection(*[set(v) for v in methods.values()])

# Column selection dropdown
col = st.selectbox("Select Column to Plot", df.columns.drop("Timestamp"))
  
# Time-series plot
st.write(f"### Time Series of {col}")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df["Timestamp"], df[col], marker=".", linestyle="-")
if col == "T_Supply":
    ax.scatter(df.loc[list(consensus),"Timestamp"],
               df.loc[list(consensus),col],
               color="red", label="Consensus Anomaly", zorder=5)
ax.set_xlabel("Timestamp")
ax.set_ylabel(col)
ax.legend()
st.pyplot(fig)

# Show anomaly detection summary for T_Supply
st.write("### T_Supply Anomaly Detection Summary")
for name, idx in methods.items():
    st.write(f"- **{name}** flagged rows: {idx.tolist()}")
st.write(f"**Consensus anomalies**: {sorted(consensus)}")

# Download processed file with anomaly flag
df["Anomaly"] = False
df.loc[list(consensus),"Anomaly"] = True
csv = df.to_csv(index=False).encode()
st.download_button("Download CSV with Anomaly Flag",
                   data=csv,
                   file_name="hvac_interactive_processed.csv",
                   mime="text/csv")
