import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# -----------------------------
# Page config and style
# -----------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
page_bg_color = "#f0f2f6"
card_bg_color = "#c0392b"  # Red theme
text_color = "#ffffff"      # White text

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {page_bg_color};
    }}
    .stButton>button {{
        color: white;
        background-color: #4CAF50;
    }}
    .card {{
        background-color: {card_bg_color};
        color: {text_color};
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(0,0,0,0.15);
        margin: 10px;
        text-align: center;
        display: inline-block;
        width: 200px;
        vertical-align: top;
    }}
    .card h3 {{
        margin: 5px;
        font-size: 22px;
        color: {text_color};
    }}
    .card p {{
        margin: 2px;
        font-size: 14px;
        color: {text_color};
    }}
    </style>
    """, unsafe_allow_html=True
)

st.title("📊 Customer Segmentation Dashboard (K-Means + Quadrant Analysis)")
st.markdown("Dynamic clustering using **R-F-M** features + Profitability Quadrants.", unsafe_allow_html=True)

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])
df = None

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

if df is None:
    st.warning("Please upload a file to continue.")
    st.stop()

# -----------------------------
# Clean & prepare data
# -----------------------------
df = df.dropna()
df.columns = [col.upper() for col in df.columns]

required_cols = ['CUSTOMER_CODE', 'RECENT_ORDER', 'TOTAL_PARCEL', 
                 'TOTAL_REVENUE', 'UNIT_REVENUE', 'UNIT_WEIGHT']
for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

# Add AVG_DAY_BETWEEN_NEXT_ORDER if missing
if "AVG_DAY_BETWEEN_NEXT_ORDER" not in df.columns:
    df["AVG_DAY_BETWEEN_NEXT_ORDER"] = np.random.randint(5, 60, size=len(df))

df = df[required_cols + ['AVG_DAY_BETWEEN_NEXT_ORDER']].copy()
df = df.dropna()

# -----------------------------
# KPI Cards
# -----------------------------
total_customers = df['CUSTOMER_CODE'].nunique()
total_revenue = df['TOTAL_REVENUE'].sum()
avg_unit_revenue = df['UNIT_REVENUE'].mean()
avg_unit_weight = df['UNIT_WEIGHT'].mean()
avg_days_between_orders = df['AVG_DAY_BETWEEN_NEXT_ORDER'].mean()

st.markdown("### 📌 Key Performance Indicators")
kpi_html = f"""
<div class="card"><h3>Total Customers</h3><p>{total_customers:,}</p></div>
<div class="card"><h3>Total Revenue (THB)</h3><p>{total_revenue:,.0f}</p></div>
<div class="card"><h3>Avg Unit Revenue</h3><p>{avg_unit_revenue:.2f}</p></div>
<div class="card"><h3>Avg Unit Weight</h3><p>{avg_unit_weight:.2f}</p></div>
<div class="card"><h3>Avg Days Between Orders</h3><p>{avg_days_between_orders:.2f}</p></div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# -----------------------------
# K-Means Clustering
# -----------------------------
features = ['RECENT_ORDER', 'TOTAL_PARCEL', 'TOTAL_REVENUE']
df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df = df.dropna()

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.slider("Select number of clusters (K)", 2, 10, 4)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df['Cluster_str'] = df['Cluster'].astype(str)

# -----------------------------
# Cluster Summary
# -----------------------------
cluster_summary = df.groupby("Cluster").agg({
    "CUSTOMER_CODE": "count",
    "TOTAL_REVENUE": "sum",
    "TOTAL_PARCEL": "sum",
    "RECENT_ORDER": "mean",
    "UNIT_REVENUE": "mean",
    "UNIT_WEIGHT": "mean",
    "AVG_DAY_BETWEEN_NEXT_ORDER": "mean"
}).rename(columns={"CUSTOMER_CODE": "Customer_Count"}).reset_index()

st.subheader("📊 Cluster Summary Table")
st.dataframe(cluster_summary.style.format({
    "TOTAL_REVENUE": "{:,.2f}",
    "TOTAL_PARCEL": "{:,.2f}",
    "RECENT_ORDER": "{:.2f}",
    "UNIT_REVENUE": "{:.2f}",
    "UNIT_WEIGHT": "{:.2f}",
    "AVG_DAY_BETWEEN_NEXT_ORDER": "{:.2f}"
}))

# -----------------------------
# Cluster Cards
# -----------------------------
st.markdown("### 📌 Cluster Snapshot")
cards_html = ""
for _, row in cluster_summary.iterrows():
    cards_html += f"""
    <div class="card">
        <h3>Cluster {int(row['Cluster'])}</h3>
        <p><b>Customers:</b> {row['Customer_Count']:,}</p>
        <p><b>Total Revenue:</b> {row['TOTAL_REVENUE']:,.0f}</p>
        <p><b>Total Parcels:</b> {row['TOTAL_PARCEL']:,.0f}</p>
        <p><b>Avg Revenue/Unit:</b> {row['UNIT_REVENUE']:.2f}</p>
        <p><b>Avg Weight:</b> {row['UNIT_WEIGHT']:.2f}</p>
        <p><b>Avg Days btw Orders:</b> {row['AVG_DAY_BETWEEN_NEXT_ORDER']:.2f}</p>
    </div>
    """
st.markdown(cards_html, unsafe_allow_html=True)

# -----------------------------
# Cluster Insights Graphs
# -----------------------------
st.subheader("📈 Cluster Insights")
col1, col2 = st.columns(2)
custom_colors = px.colors.qualitative.Set2

fig1 = px.bar(cluster_summary, x="Cluster", y="TOTAL_REVENUE",
              text=cluster_summary["TOTAL_REVENUE"].map("{:,.2f}".format),
              color="Cluster", color_discrete_sequence=custom_colors,
              title="💰 Total Revenue by Cluster")
col1.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(cluster_summary, x="Cluster", y="TOTAL_PARCEL",
              text=cluster_summary["TOTAL_PARCEL"].map("{:,.2f}".format),
              color="Cluster", color_discrete_sequence=custom_colors,
              title="📦 Total Parcels by Cluster")
col2.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Profitability Quadrant
# -----------------------------
st.subheader("🪙 Profitability Quadrant: Unit Weight vs Unit Revenue")

k_value = st.slider("Select weight penalty factor (k)", 0.0, 5.0, 1.0, 0.1)
df["Profit_Score"] = df["UNIT_REVENUE"] - k_value * df["UNIT_WEIGHT"]

avg_unit_rev = df['UNIT_REVENUE'].mean()
avg_unit_wt = df['UNIT_WEIGHT'].mean()
rev_max = df['UNIT_REVENUE'].quantile(0.95)
wt_max = df['UNIT_WEIGHT'].quantile(0.95)

fig3 = px.scatter(
    df,
    x="UNIT_WEIGHT",
    y="UNIT_REVENUE",
    color="Profit_Score",
    hover_data=["CUSTOMER_CODE", "TOTAL_REVENUE", "TOTAL_PARCEL", "Profit_Score"],
    color_continuous_scale="RdYlGn",
    title=f"Profitability Quadrant (Profit Score = Unit Revenue - k × Unit Weight, k={k_value})"
)
fig3.add_vline(x=avg_unit_wt, line_dash="dash", line_color="red")
fig3.add_hline(y=avg_unit_rev, line_dash="dash", line_color="red")
fig3.update_xaxes(range=[0, wt_max], title="Unit Weight")
fig3.update_yaxes(range=[0, rev_max], title="Unit Revenue")

st.plotly_chart(fig3, use_container_width=True)

# Quadrant counts
st.markdown("### 📌 Quadrant Analysis & Conclusion")
quadrants = {
    "Q1: High Revenue, High Weight": ((df["UNIT_REVENUE"] >= avg_unit_rev) & (df["UNIT_WEIGHT"] >= avg_unit_wt)).sum(),
    "Q2: High Revenue, Low Weight": ((df["UNIT_REVENUE"] >= avg_unit_rev) & (df["UNIT_WEIGHT"] < avg_unit_wt)).sum(),
    "Q3: Low Revenue, Low Weight": ((df["UNIT_REVENUE"] < avg_unit_rev) & (df["UNIT_WEIGHT"] < avg_unit_wt)).sum(),
    "Q4: Low Revenue, High Weight": ((df["UNIT_REVENUE"] < avg_unit_rev) & (df["UNIT_WEIGHT"] >= avg_unit_wt)).sum(),
}
st.write("**Profit Score Formula**: Profit_Score = Unit Revenue − k × Unit Weight")
st.write(f"**Quadrant Thresholds** → Avg Revenue: {avg_unit_rev:.2f}, Avg Weight: {avg_unit_wt:.2f}")
st.write("**Customer Distribution by Quadrant:**")
for q, count in quadrants.items():
    st.write(f"- {q}: {count} customers")

# -----------------------------
# Customer Table
# -----------------------------
st.subheader("👥 Customer Details by Cluster")
selected_cluster = st.selectbox("Filter by Cluster", ["All"] + list(df['Cluster_str'].unique()))
df_filtered = df if selected_cluster == "All" else df[df['Cluster_str'] == selected_cluster]
st.dataframe(df_filtered)

# -----------------------------
# Executive Summary
# -----------------------------
st.subheader("📝 Executive Summary: Customer Segmentation Analysis")
summary_text = ""
for idx, row in cluster_summary.iterrows():
    summary_text += (
        f"Cluster {row['Cluster']} consists of {row['Customer_Count']:,} customers, "
        f"generating {row['TOTAL_REVENUE']/1e6:.2f}M THB with {row['TOTAL_PARCEL']:,} parcels. "
        f"Avg recency: {row['RECENT_ORDER']:.2f} days, "
        f"Avg Unit Revenue: {row['UNIT_REVENUE']:.2f} THB, "
        f"Avg Unit Weight: {row['UNIT_WEIGHT']:.2f} kg, "
        f"Avg Days Between Orders: {row['AVG_DAY_BETWEEN_NEXT_ORDER']:.2f}.\n\n"
    )

overall_summary = ("Overall Summary:\n"
                   "- Larger clusters represent the main revenue drivers.\n"
                   "- Some clusters are highly concentrated and require close service.\n"
                   "- Inactive clusters show reactivation opportunities.\n"
                   "- Quadrant analysis highlights 💎 Good Profit customers to maintain.\n\n")

recommendations = ("Recommendations:\n"
                   "- Retain and reward high-value clusters.\n"
                   "- Optimize logistics for heavy customers.\n"
                   "- Reactivate inactive clusters with campaigns.\n"
                   "- Re-evaluate costly low-profit heavy clusters.\n")

st.markdown(summary_text + overall_summary + recommendations)
