import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_customers = 500
data = {
    'Recency': np.random.randint(1, 365, num_customers),  # Days since last purchase
    'Frequency': np.random.poisson(lam=5, size=num_customers),  # Number of purchases
    'MonetaryValue': np.random.gamma(shape=2, scale=100, size=num_customers), # Total spending
    'Churn': np.random.binomial(1, 0.2, num_customers) # 0: Not churned, 1: Churned
}
df = pd.DataFrame(data)
# --- 2. Data Preprocessing and Feature Scaling ---
# Scale the features for KMeans clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Recency', 'Frequency', 'MonetaryValue']])
# --- 3. Customer Segmentation using KMeans Clustering ---
# Determine optimal number of clusters (e.g., using the Elbow method - this is a simplification)
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_method.png')
print("Plot saved to elbow_method.png")
# Perform KMeans clustering with chosen number of clusters (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)
# --- 4. Analysis of Customer Segments ---
# Analyze the characteristics of each cluster
cluster_analysis = df.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryValue', 'Churn']].agg(['mean', 'count'])
print("\nCluster Analysis:")
print(cluster_analysis)
# --- 5. Visualization of Customer Segments ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x='MonetaryValue', y='Frequency', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Monetary Value')
plt.ylabel('Frequency')
plt.savefig('customer_segments.png')
print("Plot saved to customer_segments.png")
# --- 6. Churn Rate Analysis per Segment ---
churn_rates = df.groupby('Cluster')['Churn'].mean()
print("\nChurn Rates per Segment:")
print(churn_rates)
#Further analysis could involve more sophisticated modelling techniques to predict churn probability and lifetime value for each segment.  This example provides a basic framework.