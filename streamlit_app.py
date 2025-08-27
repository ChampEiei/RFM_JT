import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots


# -----------------------------
# Page config and style
# -----------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
page_bg_color = "#f0f2f6"
card_bg_color = "#c0392b"
text_color = "#ffffff"

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

st.title("📊 Customer Segmentation Dashboard (R-F-M + Cluster + Profit Analysis)")

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])
if uploaded_file is None:
    st.warning("Please upload a file to continue.")
    st.stop()

if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# -----------------------------
# Data cleaning
# -----------------------------
df.columns = [col.upper() for col in df.columns]

required_cols = ['CUSTOMER_CODE', 'RECENT_ORDER', 'FREQUENCY', 'TOTAL_PARCEL',
                 'TOTAL_REVENUE', 'UNIT_REVENUE', 'UNIT_WEIGHT']
for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

if "AVG_DAY_BETWEEN_NEXT_ORDER" not in df.columns:
    df["AVG_DAY_BETWEEN_NEXT_ORDER"] = np.random.randint(5, 60, size=len(df))

df = df[required_cols + ['AVG_DAY_BETWEEN_NEXT_ORDER']].copy().dropna()

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
# RFM Clustering (use FREQUENCY instead of TOTAL_PARCEL)
# -----------------------------
features = ['RECENT_ORDER', 'FREQUENCY', 'TOTAL_REVENUE']
X = df[features].apply(pd.to_numeric, errors='coerce').dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.slider("Select number of clusters (K)", 2, 10, 4)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df['Cluster_str'] = df['Cluster'].astype(str)

# -----------------------------
# Cluster summary table
# -----------------------------
cluster_summary = df.groupby("Cluster").agg({
    "CUSTOMER_CODE": "count",
    "TOTAL_REVENUE": "sum",
    "FREQUENCY": "mean",
    "TOTAL_PARCEL": "sum",
    "RECENT_ORDER": "mean",
    "UNIT_REVENUE": "mean",
    "UNIT_WEIGHT": "mean",
    "AVG_DAY_BETWEEN_NEXT_ORDER": "mean"
}).rename(columns={"CUSTOMER_CODE": "Customer_Count"}).reset_index()

st.subheader("📊 Cluster Summary Table")
st.dataframe(cluster_summary.style.format({
    "TOTAL_REVENUE": "{:,.2f}",
    "FREQUENCY": "{:.2f}",
    "TOTAL_PARCEL": "{:,.2f}",
    "RECENT_ORDER": "{:.2f}",
    "UNIT_REVENUE": "{:.2f}",
    "UNIT_WEIGHT": "{:.2f}",
    "AVG_DAY_BETWEEN_NEXT_ORDER": "{:.2f}"
}))

# -----------------------------
import matplotlib.pyplot as plt

st.subheader("🔹 2D Scatter Plot Matrix (Static): R-F-M by Cluster")

features = ["RECENT_ORDER", "FREQUENCY", "TOTAL_REVENUE"]
clusters = sorted(df["Cluster"].unique())

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
colors = plt.cm.tab10.colors  # Professional categorical palette

for i, y_feat in enumerate(features):
    for j, x_feat in enumerate(features):
        ax = axes[i, j]
        for cluster in clusters:
            cluster_df = df[df["Cluster"] == cluster]
            ax.scatter(
                cluster_df[x_feat],
                cluster_df[y_feat],
                s=20,
                color=colors[cluster % len(colors)],
                label=f"Cluster {cluster}" if (i == 0 and j == 0) else None,
                alpha=0.7
            )
        if i == 2:
            ax.set_xlabel(x_feat)
        else:
            ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel(y_feat)
        else:
            ax.set_ylabel("")

fig.suptitle("2D R-F-M Scatter Plot Matrix by Cluster", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.legend(title="Clusters", loc="upper right")

st.pyplot(fig)



# -----------------------------
# Violin plots per cluster
# -----------------------------
st.subheader("📊 Cluster Distribution by R, F, M (Horizontal Layout)")

features = ['RECENT_ORDER', 'FREQUENCY', 'TOTAL_REVENUE']
cols = st.columns(3)

for i, feature in enumerate(features):
    with cols[i]:
        fig_vio = go.Figure()
        for cluster in sorted(df['Cluster'].unique()):
            cluster_df = df[df['Cluster'] == cluster]
            fig_vio.add_trace(go.Violin(
                x=[str(cluster)] * len(cluster_df),
                y=cluster_df[feature],
                name=f"Cluster {cluster}",
                box_visible=True,
                meanline_visible=True,
                line_color=f"rgba({cluster*30 % 255}, {cluster*60 % 255}, {cluster*90 % 255}, 1)"
            ))
        fig_vio.update_layout(
            title=f"{feature} Distribution by Cluster",
            xaxis_title="Cluster",
            yaxis_title=feature,
            violingap=0,
            violinmode='group'
        )
        st.plotly_chart(fig_vio, use_container_width=True)

# -----------------------------
# Weight vs Unit Revenue with Linear Regression + Profit Score
# -----------------------------
st.subheader("⚖️ Weight vs Unit Revenue Analysis")
k_value = st.slider("Profit Score Penalty (k)", 0.0, 5.0, 1.0, 0.1)
coeff_slider = st.slider("Linear Coefficient for regression line", 0.0, 5.0, 1.0, 0.05)
df['Profit_Score'] = df['UNIT_REVENUE'] - k_value * df['UNIT_WEIGHT']

# Linear Regression
lr = LinearRegression()
lr.fit(df[['UNIT_WEIGHT']], df['UNIT_REVENUE'])
df['Linear_Pred'] = lr.predict(df[['UNIT_WEIGHT']]) * coeff_slider

fig_wv = px.scatter(df, x='UNIT_WEIGHT', y='UNIT_REVENUE', color='Profit_Score',
                    color_continuous_scale='RdYlGn',
                    hover_data=['CUSTOMER_CODE', 'TOTAL_REVENUE', 'FREQUENCY'])
fig_wv.add_hline(y=df['UNIT_REVENUE'].mean(), line_dash="dash", line_color="red")
fig_wv.add_vline(x=df['UNIT_WEIGHT'].mean(), line_dash="dash", line_color="red")
fig_wv.add_trace(go.Scatter(x=df['UNIT_WEIGHT'], y=df['Linear_Pred'],
                            mode='lines', name='Linear Fit', line=dict(color='blue', width=2)))
st.plotly_chart(fig_wv, use_container_width=True)

# -----------------------------
# Comparison Table for Q4 + Linear Below Line
# -----------------------------
st.subheader("📋 Comparison Table: Unit Revenue Adjustment Methods (Aggregated)")

# Method 1: Quadrant 4 adjustment
q4_df = df[(df['UNIT_REVENUE'] < df['UNIT_REVENUE'].mean()) & 
           (df['UNIT_WEIGHT'] >= df['UNIT_WEIGHT'].mean())].copy()
q4_new_unit_rev = (q4_df['UNIT_REVENUE'] * 1.1).mean()  # Example: adjusted mean
q4_total_revenue_new = (q4_df['TOTAL_PARCEL'] * q4_df['UNIT_REVENUE'] * 1.1).sum()
q4_total_revenue_old = q4_df['TOTAL_REVENUE'].sum()
q4_profit_change = q4_total_revenue_new - q4_total_revenue_old
q4_avg_weight = q4_df['UNIT_WEIGHT'].mean()

# Method 2: Linear Fit adjustment
under_line_df = df[df['UNIT_REVENUE'] < df['Linear_Pred']].copy()
linear_new_unit_rev = under_line_df['Linear_Pred'].mean()
linear_total_revenue_new = (under_line_df['TOTAL_PARCEL'] * under_line_df['Linear_Pred']).sum()
linear_total_revenue_old = under_line_df['TOTAL_REVENUE'].sum()
linear_profit_change = linear_total_revenue_new - linear_total_revenue_old
linear_avg_weight = under_line_df['UNIT_WEIGHT'].mean()

# Create aggregated comparison table
comparison_df = pd.DataFrame({
    "Method": ["Q4_Adjust", "Linear_Adjust"],
    "Old_Unit_Revenue": [q4_df['UNIT_REVENUE'].mean(), under_line_df['UNIT_REVENUE'].mean()],
    "New_Unit_Revenue": [q4_new_unit_rev, linear_new_unit_rev],
    "Unit_Weight": [q4_avg_weight, linear_avg_weight],
    "Total_Revenue_Before": [q4_total_revenue_old, linear_total_revenue_old],
    "Total_Revenue_After": [q4_total_revenue_new, linear_total_revenue_new],
    "Profit_Change": [q4_profit_change, linear_profit_change]
})

st.dataframe(comparison_df.style.format({
    "Old_Unit_Revenue": "{:.2f}",
    "New_Unit_Revenue": "{:.2f}",
    "Unit_Weight": "{:.2f}",
    "Total_Revenue_Before": "{:,.0f}",
    "Total_Revenue_After": "{:,.0f}",
    "Profit_Change": "{:,.0f}"
}))


# -----------------------------
# Customer Table Filtered
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
        f"Avg Frequency: {row['FREQUENCY']:.2f}, "
        f"Avg Unit Revenue: {row['UNIT_REVENUE']:.2f} THB, "
        f"Avg Unit Weight: {row['UNIT_WEIGHT']:.2f} kg, "
        f"Avg Days Between Orders: {row['AVG_DAY_BETWEEN_NEXT_ORDER']:.2f}.\n\n"
    )

overall_summary = ("Overall Summary:\n"
                   "- RFM clustering (Recency, Frequency, Monetary) identifies customer segments and revenue drivers.\n"
                   "- Weight vs Unit Revenue analysis shows strategic pricing opportunities.\n"
                   "- Quadrant 4 and Linear adjustments provide targeted profit improvement actions.\n\n")
recommendations = ("Recommendations:\n"
                   "- Retain high-value clusters.\n"
                   "- Adjust pricing for Quadrant 4 customers.\n"
                   "- Monitor underperforming customers below linear trend for upsell opportunities.\n")

st.markdown(summary_text + overall_summary + recommendations)
