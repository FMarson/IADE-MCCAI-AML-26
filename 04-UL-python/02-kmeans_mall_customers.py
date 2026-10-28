# Required packages:
# pip install pandas numpy matplotlib seaborn scikit-learn

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

# Improve the appearance of plots
sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------
print("\n[1/9] Loading the dataset...")

file_path = Path("mall_customers.csv")

if not file_path.exists():
    raise FileNotFoundError(
        f"Could not find '{file_path}'. "
        "Place the CSV file in the same folder as this script."
    )

df = pd.read_csv(file_path)

print(f"Dataset loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns")
print("\nFirst five rows:")
print(df.head())


# ------------------------------------------------------------
# 2. Inspect and clean the data
# ------------------------------------------------------------
print("\n[2/9] Inspecting and cleaning the data...")

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types before cleaning:")
print(df.dtypes)

# The attached file uses "Genre" instead of "Gender".
# We will rename it to a more commonly used name.
if "Genre" in df.columns:
    df = df.rename(columns={"Genre": "Gender"})

# Convert numeric columns explicitly.
numeric_columns = [
    "CustomerID",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows with missing values in required columns.
required_columns = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

missing_values = df[required_columns].isna().sum()

print("\nMissing values:")
print(missing_values)

df = df.dropna(subset=required_columns).copy()
df = df.reset_index(drop=True)

print(f"\nRows after cleaning: {len(df)}")

if len(df) == 0:
    raise ValueError("The dataset contains no usable rows after cleaning.")


# ------------------------------------------------------------
# 3. Explore the data
# ------------------------------------------------------------
print("\n[3/9] Exploring the data...")

print("\nDescriptive statistics:")
print(df.describe())

print("\nGender distribution:")
print(df["Gender"].value_counts())

print("\nBasic visual exploration...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.histplot(
    data=df,
    x="Annual Income (k$)",
    bins=15,
    kde=True,
    ax=axes[0],
    color="steelblue"
)

axes[0].set_title("Annual Income Distribution")
axes[0].set_xlabel("Annual Income (k$)")

sns.histplot(
    data=df,
    x="Spending Score (1-100)",
    bins=15,
    kde=True,
    ax=axes[1],
    color="darkorange"
)

axes[1].set_title("Spending Score Distribution")
axes[1].set_xlabel("Spending Score (1-100)")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 4. Select the features for clustering
# ------------------------------------------------------------
print("\n[4/9] Selecting features for clustering...")

# These are the two most commonly used features for this dataset.
feature_columns = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[feature_columns].copy()

print("\nFeatures used by K-Means:")
print(feature_columns)

print("\nFeature preview:")
print(X.head())


# ------------------------------------------------------------
# 5. Scale the features
# ------------------------------------------------------------
print("\n[5/9] Scaling the features...")

# Standardization gives each feature mean 0 and standard deviation 1.
# This prevents one feature from having excessive influence because
# of its numerical scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features scaled successfully.")
print(f"Scaled data shape: {X_scaled.shape}")


# ------------------------------------------------------------
# 6. Compare possible numbers of clusters
# ------------------------------------------------------------
print("\n[6/9] Testing different numbers of clusters...")

inertias = []
silhouette_scores = []

k_values = range(2, 11)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )

results = pd.DataFrame({
    "k": list(k_values),
    "Inertia": inertias,
    "Silhouette Score": silhouette_scores
})

print("\nEvaluation results:")
print(results.to_string(index=False))

best_k = int(
    results.loc[
        results["Silhouette Score"].idxmax(),
        "k"
    ]
)

best_silhouette = results["Silhouette Score"].max()

print(f"\nBest k according to the silhouette score: {best_k}")
print(f"Best silhouette score: {best_silhouette:.4f}")


# ------------------------------------------------------------
# 7. Plot the elbow and silhouette results
# ------------------------------------------------------------
print("\n[7/9] Plotting model evaluation results...")

fig, ax1 = plt.subplots(figsize=(9, 5))

# Inertia axis
line1 = ax1.plot(
    list(k_values),
    inertias,
    marker="o",
    color="blue",
    linewidth=2,
    label="Inertia"
)

ax1.set_xlabel("Number of clusters, k")
ax1.set_ylabel("Inertia", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.set_xticks(list(k_values))

# Silhouette axis
ax2 = ax1.twinx()

line2 = ax2.plot(
    list(k_values),
    silhouette_scores,
    marker="o",
    color="green",
    linewidth=2,
    label="Silhouette score"
)

ax2.set_ylabel("Silhouette score", color="green")
ax2.tick_params(axis="y", labelcolor="green")
ax2.set_ylim(0, 1)

# Highlight the best silhouette score
ax2.scatter(
    best_k,
    best_silhouette,
    color="red",
    s=100,
    zorder=5,
    label=f"Best k = {best_k}"
)

# Combine legends from both axes
lines = line1 + line2
labels = [line.get_label() for line in lines]

extra_line = plt.Line2D(
    [],
    [],
    color="red",
    marker="o",
    linestyle="None",
    markersize=8,
    label=f"Best k = {best_k}"
)

ax1.legend(
    lines + [extra_line],
    labels + [f"Best k = {best_k}"],
    loc="center right"
)

plt.title("Choosing the Number of Clusters")
fig.tight_layout()

plt.savefig(
    "01_model_evaluation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 8. Train and visualize the final K-Means model
# ------------------------------------------------------------
print("\n[8/9] Training and visualizing the final K-Means model...")

# Use the k selected by the highest silhouette score.
final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = final_kmeans.fit_predict(X_scaled)

# Cluster centers are currently in standardized coordinates.
# Convert them back to the original income and score units.
cluster_centers = scaler.inverse_transform(
    final_kmeans.cluster_centers_
)

centers_df = pd.DataFrame(
    cluster_centers,
    columns=feature_columns
)

centers_df.index.name = "Cluster"

print(f"\nFinal model trained with k = {best_k}")

print("\nCluster centers in original units:")
print(centers_df.round(2))

# Plot the clusters using the original feature values.
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=100,
    alpha=0.85
)

plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    marker="X",
    s=300,
    color="black",
    edgecolor="white",
    linewidth=1.5,
    label="Cluster centers"
)

plt.title(f"Customer Clusters Using K-Means, k = {best_k}")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.tight_layout()

plt.savefig(
    "02_customer_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 9. Analyze and save the customer segments
# ------------------------------------------------------------
print("\n[9/9] Analyzing and saving the customer segments...")

cluster_summary = (
    df.groupby("Cluster")
      .agg(
          Customers=("CustomerID", "count"),
          Average_Age=("Age", "mean"),
          Average_Income=("Annual Income (k$)", "mean"),
          Average_Spending_Score=("Spending Score (1-100)", "mean")
      )
      .round(2)
      .sort_index()
)

print("\nCluster summary:")
print(cluster_summary)

print("\nCluster sizes:")
print(df["Cluster"].value_counts().sort_index())

# Give each cluster a simple descriptive label.
# The labels are based on average income and spending score.
income_median = df["Annual Income (k$)"].median()
score_median = df["Spending Score (1-100)"].median()

def describe_cluster(row):
    income = row["Average_Income"]
    score = row["Average_Spending_Score"]

    if income >= income_median and score >= score_median:
        return "High income / high spending"
    elif income >= income_median and score < score_median:
        return "High income / low spending"
    elif income < income_median and score >= score_median:
        return "Low income / high spending"
    else:
        return "Low income / low spending"

cluster_summary["Description"] = cluster_summary.apply(
    describe_cluster,
    axis=1
)

print("\nCluster descriptions:")
print(cluster_summary)

# Add descriptions to each customer record.
description_by_cluster = cluster_summary["Description"].to_dict()
df["Cluster Description"] = df["Cluster"].map(description_by_cluster)

# Save the labeled customer data.
df.to_csv(
    "mall_customers_with_clusters.csv",
    index=False
)

# Save the cluster summary.
cluster_summary.to_csv(
    "cluster_summary.csv"
)

print("\nFiles saved:")
print("- 01_model_evaluation.png")
print("- 02_customer_clusters.png")
print("- mall_customers_with_clusters.csv")
print("- cluster_summary.csv")

print("\nK-Means analysis completed successfully.")
