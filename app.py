#%%

import pandas as pd 
from  sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#%%

df = pd.read_excel(r'raw_data\RFM_model_ june to 24 august.xlsx')
df.columns

#%%
df = df[~df.isnull().any(axis=1)]
df.columns

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Suppose your DataFrame is df
# Columns: CUSTOMER_CODE, AVG_DAY_BETWEEN_NEXT_ORDER, RECENT_ORDER, TOTAL_PARCEL, TOTAL_REVENUE, RANK_CUS

# --- Step 1: Build RFM features ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


# --- Step 1: Build RFM features ---
rfm = df[['CUSTOMER_CODE', 'RECENT_ORDER', 'TOTAL_PARCEL', 'TOTAL_REVENUE']].copy()
rfm = rfm.rename(columns={
    'RECENT_ORDER': 'Recency',
    'TOTAL_PARCEL': 'Frequency',
    'TOTAL_REVENUE': 'Monetary'
})

# --- Step 2: Scale values ---
X = rfm[['Recency','Frequency','Monetary']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 3: Elbow method to find optimal k ---
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method For Optimal k')
plt.show()

# --- Step 4: Fit final model ---
optimal_k = 4  # adjust based on elbow
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Step 5: 3D Scatter Plot ---
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    rfm['Recency'], rfm['Frequency'], rfm['Monetary'],
    c=rfm['Cluster'], cmap='viridis', s=50
)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.title("RFM Clustering - 3D Scatter")
plt.colorbar(sc)
plt.show()

# --- Step 6: Aggregate analysis per cluster ---
cluster_analysis = rfm.groupby('Cluster').agg({
    'CUSTOMER_CODE': 'count',        # Number of customers
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['sum','mean']       # Total sales, Avg Monetary
}).reset_index()

# Flatten MultiIndex columns
cluster_analysis.columns = ['Cluster','Num_Customers','Avg_Recency','Avg_Frequency','Total_Monetary','Avg_Monetary']

print(cluster_analysis)

# --- Step 7: Plot cluster sales and parcels ---
fig, ax = plt.subplots(1,2, figsize=(14,6))

# Total Monetary per cluster
ax[0].bar(cluster_analysis['Cluster'], cluster_analysis['Total_Monetary'], color='skyblue')
ax[0].set_xlabel('Cluster')
ax[0].set_ylabel('Total Revenue')
ax[0].set_title('Total Revenue per Cluster')

# Avg Frequency per cluster
ax[1].bar(cluster_analysis['Cluster'], cluster_analysis['Avg_Frequency'], color='salmon')
ax[1].set_xlabel('Cluster')
ax[1].set_ylabel('Average Frequency')
ax[1].set_title('Average Frequency per Cluster')

plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# --- Step 1: Build RFM features ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# --- Step 1: Build RFM features with AVG_DAY_BETWEEN_NEXT_ORDER as 'Activity' ---
rfm = df[['CUSTOMER_CODE', 'RECENT_ORDER', 'TOTAL_PARCEL', 'TOTAL_REVENUE', 'AVG_DAY_BETWEEN_NEXT_ORDER']].copy()
rfm = rfm.rename(columns={
    'RECENT_ORDER': 'Recency',
    'TOTAL_PARCEL': 'Frequency',
    'TOTAL_REVENUE': 'Monetary',
    'AVG_DAY_BETWEEN_NEXT_ORDER': 'Avg_Order_Interval'
})

# Convert Avg_Order_Interval to activity (smaller interval -> higher activity)
rfm['Activity'] = 1 / (rfm['Avg_Order_Interval'] + 1)  # +1 to avoid divide by zero

# --- Step 2: Log-transform skewed features ---
rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
rfm['Activity_log'] = np.log1p(rfm['Activity'])

# --- Step 3: Scale features using RobustScaler ---
X = rfm[['Recency', 'Frequency_log', 'Monetary_log', 'Activity_log']]
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 4: Elbow method to choose optimal k ---
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method For Optimal k')
plt.show()

# --- Step 5: Fit final model (choose k from elbow, e.g., k=4) ---
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Step 6: 3D scatter plot (Recency vs Frequency vs Monetary) ---
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    rfm['Recency'], rfm['Frequency_log'], rfm['Monetary_log'],
    c=rfm['Cluster'], cmap='viridis', s=50
)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency (log)')
ax.set_zlabel('Monetary (log)')
plt.title("RFM Clustering - 3D Scatter (Log-transformed + Activity)")
plt.colorbar(sc)
plt.show()

# --- Step 7: Aggregate analysis per cluster ---
cluster_analysis = rfm.groupby('Cluster').agg({
    'CUSTOMER_CODE': 'count',
    'Recency': 'mean',
    'Frequency': ['mean','sum'],
    'Monetary': ['mean','sum'],
    'Activity': 'mean'
}).reset_index()

# Flatten MultiIndex columns
cluster_analysis.columns = ['Cluster', 'Num_Customers', 'Avg_Recency', 
                            'Avg_Frequency', 'Total_Frequency',
                            'Avg_Monetary', 'Total_Monetary',
                            'Avg_Activity']

print(cluster_analysis)

# --- Step 8: Bar plots for cluster insights ---
fig, ax = plt.subplots(1,3, figsize=(18,6))

ax[0].bar(cluster_analysis['Cluster'], cluster_analysis['Total_Monetary'], color='skyblue')
ax[0].set_xlabel('Cluster')
ax[0].set_ylabel('Total Revenue')
ax[0].set_title('Total Revenue per Cluster')

ax[1].bar(cluster_analysis['Cluster'], cluster_analysis['Avg_Frequency'], color='salmon')
ax[1].set_xlabel('Cluster')
ax[1].set_ylabel('Average Frequency')
ax[1].set_title('Average Frequency per Cluster')

ax[2].bar(cluster_analysis['Cluster'], cluster_analysis['Avg_Activity'], color='lightgreen')
ax[2].set_xlabel('Cluster')
ax[2].set_ylabel('Average Activity')
ax[2].set_title('Average Activity per Cluster')

plt.show()

# --- Step 9: Optional Stage 2 Clustering on majority cluster ---
major_cluster_id = rfm['Cluster'].value_counts().idxmax()
major_cluster = rfm[rfm['Cluster'] == major_cluster_id].copy()
X_major = major_cluster[['Recency','Frequency_log','Monetary_log','Activity_log']]
X_major_scaled = scaler.fit_transform(X_major)

# Example: split majority into 3 sub-clusters
kmeans_major = KMeans(n_clusters=3, random_state=42)
major_cluster['SubCluster'] = kmeans_major.fit_predict(X_major_scaled)

# Inspect sub-clusters
sub_cluster_analysis = major_cluster.groupby('SubCluster').agg({
    'CUSTOMER_CODE':'count',
    'Recency':'mean',
    'Frequency':['mean','sum'],
    'Monetary':['mean','sum'],
    'Activity':'mean'
}).reset_index()

# Flatten columns
sub_cluster_analysis.columns = ['SubCluster','Num_Customers','Avg_Recency','Avg_Frequency','Total_Frequency','Avg_Monetary','Total_Monetary','Avg_Activity']
print(sub_cluster_analysis)

#%%
import polars as pl
pl.from_pandas(rfm).filter(pl.col('Cluster')==0)


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# --- Step 1: Build features ---
rfm = df[['CUSTOMER_CODE', 'RECENT_ORDER', 'AVG_DAY_BETWEEN_NEXT_ORDER', 
          'TOTAL_PARCEL', 'TOTAL_REVENUE', 'UNIT_REVENUE', 'UNIT_WEIGHT']].copy()

# --- Step 2: Log-transform skewed features ---
rfm['Frequency_log'] = np.log1p(rfm['TOTAL_PARCEL'])
rfm['Monetary_log'] = np.log1p(rfm['TOTAL_REVENUE'])
rfm['UnitRevenue_log'] = np.log1p(rfm['UNIT_REVENUE'])
rfm['UnitWeight_log'] = np.log1p(rfm['UNIT_WEIGHT'])
rfm['Activity_log'] = np.log1p(rfm['AVG_DAY_BETWEEN_NEXT_ORDER'])

# --- Step 3: Scale features using RobustScaler ---
features = ['RECENT_ORDER', 'Activity_log', 'Frequency_log', 'Monetary_log', 'UnitRevenue_log', 'UnitWeight_log']
scaler = RobustScaler()
X_scaled = scaler.fit_transform(rfm[features])

# --- Step 4: Elbow method to find optimal k ---
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method For Optimal k')
plt.show()

# --- Step 5: Fit final KMeans (choose optimal k, e.g., 4) ---
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Step 6: 3D Scatter Plot (using Recency, Frequency, Monetary) ---
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    rfm['RECENT_ORDER'], rfm['Frequency_log'], rfm['Monetary_log'],
    c=rfm['Cluster'], cmap='viridis', s=50
)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency (log)')
ax.set_zlabel('Monetary (log)')
plt.title("RFM + Unit Features Clustering - 3D Scatter")
plt.colorbar(sc)
plt.show()

# --- Step 7: Cluster aggregation ---
cluster_analysis = rfm.groupby('Cluster').agg({
    'CUSTOMER_CODE': 'count',
    'RECENT_ORDER': 'mean',
    'AVG_DAY_BETWEEN_NEXT_ORDER': 'mean',
    'TOTAL_PARCEL': ['mean','sum'],
    'TOTAL_REVENUE': ['mean','sum'],
    'UNIT_REVENUE': 'mean',
    'UNIT_WEIGHT': 'mean'
}).reset_index()

cluster_analysis.columns = ['Cluster','Num_Customers','Avg_Recency','Avg_Activity',
                            'Avg_Frequency','Total_Frequency','Avg_Monetary','Total_Monetary',
                            'Avg_UnitRevenue','Avg_UnitWeight']

print(cluster_analysis)

# --- Step 8: Optional Stage 2: Sub-cluster on majority cluster ---
major_cluster_id = rfm['Cluster'].value_counts().idxmax()
major_cluster = rfm[rfm['Cluster']==major_cluster_id].copy()
X_major = scaler.fit_transform(major_cluster[features])

kmeans_major = KMeans(n_clusters=3, random_state=42)
major_cluster['SubCluster'] = kmeans_major.fit_predict(X_major)

sub_cluster_analysis = major_cluster.groupby('SubCluster').agg({
    'CUSTOMER_CODE':'count',
    'RECENT_ORDER':'mean',
    'AVG_DAY_BETWEEN_NEXT_ORDER':'mean',
    'TOTAL_PARCEL':['mean','sum'],
    'TOTAL_REVENUE':['mean','sum'],
    'UNIT_REVENUE':'mean',
    'UNIT_WEIGHT':'mean'
}).reset_index()

sub_cluster_analysis.columns = ['SubCluster','Num_Customers','Avg_Recency','Avg_Activity',
                                'Avg_Frequency','Total_Frequency','Avg_Monetary','Total_Monetary',
                                'Avg_UnitRevenue','Avg_UnitWeight']

print(sub_cluster_analysis)

#%%
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Customer Segmentation (RFM + Unit Features)")
st.markdown("### Clustering using KMeans with RFM + UnitRevenue + UnitWeight")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Preview Data")
    st.dataframe(df.head())

    # --- Step 1: Select RFM features ---
    rfm = df[['CUSTOMER_CODE', 'RECENT_ORDER', 'AVG_DAY_BETWEEN_NEXT_ORDER',
              'TOTAL_PARCEL', 'TOTAL_REVENUE', 'UNIT_REVENUE', 'UNIT_WEIGHT']].copy()

    # --- Step 2: Log-transform skewed features ---
    rfm['Frequency_log'] = np.log1p(rfm['TOTAL_PARCEL'])
    rfm['Monetary_log'] = np.log1p(rfm['TOTAL_REVENUE'])
    rfm['UnitRevenue_log'] = np.log1p(rfm['UNIT_REVENUE'])
    rfm['UnitWeight_log'] = np.log1p(rfm['UNIT_WEIGHT'])
    rfm['Activity_log'] = np.log1p(rfm['AVG_DAY_BETWEEN_NEXT_ORDER'])

    # --- Step 3: Scale features ---
    features = ['RECENT_ORDER','Activity_log','Frequency_log','Monetary_log',
                'UnitRevenue_log','UnitWeight_log']
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(rfm[features])

    # --- Step 4: Elbow method ---
    st.subheader("Elbow Method")
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K, inertia, 'bo-')
    ax.set_xlabel('Number of clusters k')
    ax.set_ylabel('Inertia (SSE)')
    ax.set_title('Elbow Method For Optimal k')
    st.pyplot(fig)

    # --- Step 5: Choose cluster ---
    optimal_k = st.slider("Select number of clusters (k)", 2, 10, 4)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(X_scaled)

    # --- Step 6: 3D Scatter Plot ---
    st.subheader("3D Scatter Plot")
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(rfm['RECENT_ORDER'], rfm['Frequency_log'], rfm['Monetary_log'],
                    c=rfm['Cluster'], cmap='viridis', s=40)
    ax.set_xlabel("Recency")
    ax.set_ylabel("Frequency (log)")
    ax.set_zlabel("Monetary (log)")
    plt.colorbar(sc)
    st.pyplot(fig)

    # --- Step 7: Cluster aggregation ---
    cluster_analysis = rfm.groupby('Cluster').agg({
        'CUSTOMER_CODE': 'count',
        'RECENT_ORDER': 'mean',
        'AVG_DAY_BETWEEN_NEXT_ORDER': 'mean',
        'TOTAL_PARCEL': ['mean','sum'],
        'TOTAL_REVENUE': ['mean','sum'],
        'UNIT_REVENUE': 'mean',
        'UNIT_WEIGHT': 'mean'
    }).reset_index()

    cluster_analysis.columns = ['Cluster','Num_Customers','Avg_Recency','Avg_Activity',
                                'Avg_Frequency','Total_Frequency','Avg_Monetary','Total_Monetary',
                                'Avg_UnitRevenue','Avg_UnitWeight']

    st.subheader("Cluster Analysis")
    st.dataframe(cluster_analysis)

    # --- Step 8: Sub-cluster on majority cluster ---
    st.subheader("Sub-clustering on Majority Cluster")
    major_cluster_id = rfm['Cluster'].value_counts().idxmax()
    st.write(f"Majority Cluster: **{major_cluster_id}**")

    major_cluster = rfm[rfm['Cluster']==major_cluster_id].copy()
    X_major = scaler.fit_transform(major_cluster[features])

    kmeans_major = KMeans(n_clusters=3, random_state=42)
    major_cluster['SubCluster'] = kmeans_major.fit_predict(X_major)

    sub_cluster_analysis = major_cluster.groupby('SubCluster').agg({
        'CUSTOMER_CODE':'count',
        'RECENT_ORDER':'mean',
        'AVG_DAY_BETWEEN_NEXT_ORDER':'mean',
        'TOTAL_PARCEL':['mean','sum'],
        'TOTAL_REVENUE':['mean','sum'],
        'UNIT_REVENUE':'mean',
        'UNIT_WEIGHT':'mean'
    }).reset_index()

    sub_cluster_analysis.columns = ['SubCluster','Num_Customers','Avg_Recency','Avg_Activity',
                                    'Avg_Frequency','Total_Frequency','Avg_Monetary','Total_Monetary',
                                    'Avg_UnitRevenue','Avg_UnitWeight']

    st.dataframe(sub_cluster_analysis)

else:
    st.info("Please upload a CSV file to start clustering.")


