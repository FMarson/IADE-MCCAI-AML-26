# Required packages:
# python -m pip install -r requirements.txt

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# 1. Define project directories
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures" / "03"
EXTRAS_DIR = BASE_DIR / "extras" / "03"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EXTRAS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. Load the dataset
# ------------------------------------------------------------
print("\n[1/9] Loading the dataset...")

file_path = DATA_DIR / "mall_customers.csv"

if not file_path.exists():
    raise FileNotFoundError(
        f"Could not find '{file_path}'. "
        "Place 'mall_customers.csv' in the 'data' folder."
    )

df = pd.read_csv(file_path)

print(
    f"Dataset loaded successfully: "
    f"{df.shape[0]} rows and {df.shape[1]} columns"
)

print("\nFirst five rows:")
print(df.head())


# ------------------------------------------------------------
# 3. Inspect and clean the data
# ------------------------------------------------------------
print("\n[2/9] Inspecting and cleaning the data...")

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types before cleaning:")
print(df.dtypes)

# The original dataset uses "Genre".
# Rename it to "Gender".
if "Genre" in df.columns:
    df = df.rename(
        columns={
            "Genre": "Gender"
        }
    )

numeric_columns = [
    "CustomerID",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

required_columns = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

print("\nMissing values before cleaning:")
print(
    df[required_columns].isna().sum()
)

df = df.dropna(
    subset=required_columns
).copy()

df = df.reset_index(
    drop=True
)

if len(df) == 0:
    raise ValueError(
        "No usable rows remain after cleaning."
    )

print(
    f"\nRows after cleaning: {len(df)}"
)


# ------------------------------------------------------------
# 4. Select and scale the features
# ------------------------------------------------------------
print(
    "\n[3/9] Selecting and scaling the features..."
)

feature_columns = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[feature_columns].copy()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Features used:")
print(feature_columns)

print(
    f"\nScaled data shape: {X_scaled.shape}"
)


# ------------------------------------------------------------
# 5. Plot the dendrogram
# ------------------------------------------------------------
print(
    "\n[4/9] Creating the dendrogram..."
)

linkage_matrix = linkage(
    X_scaled,
    method="ward",
    metric="euclidean"
)

plt.figure(
    figsize=(14, 7)
)

dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)

plt.title(
    "Hierarchical Clustering Dendrogram"
)

plt.xlabel(
    "Groups of customers"
)

plt.ylabel(
    "Ward distance"
)

plt.tight_layout()

dendrogram_file = (
    FIGURE_DIR / "01_dendrogram.png"
)

plt.savefig(
    dendrogram_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 6. Compare possible numbers of clusters
# ------------------------------------------------------------
print(
    "\n[5/9] Testing different numbers of clusters..."
)

cluster_values = range(2, 11)
silhouette_scores = []

for n_clusters in cluster_values:
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage="ward"
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(
        score
    )

results = pd.DataFrame({
    "n_clusters": list(cluster_values),
    "Silhouette Score": silhouette_scores
})

print("\nEvaluation results:")
print(
    results.to_string(
        index=False
    )
)

best_n_clusters = int(
    results.loc[
        results["Silhouette Score"].idxmax(),
        "n_clusters"
    ]
)

best_silhouette = results[
    "Silhouette Score"
].max()

print(
    f"\nBest number of clusters "
    f"according to silhouette score: "
    f"{best_n_clusters}"
)

print(
    f"Best silhouette score: "
    f"{best_silhouette:.4f}"
)


# ------------------------------------------------------------
# 7. Plot silhouette scores
# ------------------------------------------------------------
print(
    "\n[6/9] Plotting silhouette scores..."
)

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    list(cluster_values),
    silhouette_scores,
    marker="o",
    linewidth=2,
    color="darkgreen"
)

plt.scatter(
    best_n_clusters,
    best_silhouette,
    color="red",
    s=120,
    zorder=5,
    label=f"Best value = {best_n_clusters}"
)

plt.title(
    "Silhouette Score by Number of Clusters"
)

plt.xlabel(
    "Number of clusters"
)

plt.ylabel(
    "Silhouette score"
)

plt.xticks(
    list(cluster_values)
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.tight_layout()

silhouette_file = (
    FIGURE_DIR / "02_silhouette_scores.png"
)

plt.savefig(
    silhouette_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 8. Train the final model
# ------------------------------------------------------------
print(
    "\n[7/9] Training the final "
    "Agglomerative Clustering model..."
)

final_model = AgglomerativeClustering(
    n_clusters=best_n_clusters,
    metric="euclidean",
    linkage="ward"
)

df["Cluster"] = final_model.fit_predict(
    X_scaled
)

print(
    f"\nFinal model trained with "
    f"{best_n_clusters} clusters."
)


# ------------------------------------------------------------
# 9. Visualize the customer segments
# ------------------------------------------------------------
print(
    "\n[8/9] Visualizing customer segments..."
)

plt.figure(
    figsize=(9, 6)
)

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=100,
    alpha=0.85
)

plt.title(
    "Customer Segments Using "
    f"Agglomerative Clustering, "
    f"n = {best_n_clusters}"
)

plt.xlabel(
    "Annual Income (k$)"
)

plt.ylabel(
    "Spending Score (1-100)"
)

plt.legend(
    title="Cluster"
)

plt.tight_layout()

clusters_file = (
    FIGURE_DIR / "03_agglomerative_clusters.png"
)

plt.savefig(
    clusters_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 10. Create the cluster summary
# ------------------------------------------------------------
print(
    "\n[9/9] Creating the cluster summary..."
)

cluster_summary = (
    df.groupby("Cluster")
      .agg(
          Customers=("CustomerID", "count"),
          Average_Age=("Age", "mean"),
          Average_Income=(
              "Annual Income (k$)",
              "mean"
          ),
          Average_Spending_Score=(
              "Spending Score (1-100)",
              "mean"
          )
      )
      .round(2)
      .sort_index()
)

print("\nCluster summary:")
print(cluster_summary)

print("\nCluster sizes:")
print(
    df["Cluster"].value_counts().sort_index()
)


# ------------------------------------------------------------
# 11. Add descriptive labels
# ------------------------------------------------------------
print(
    "\nAdding descriptive cluster labels..."
)

income_median = df[
    "Annual Income (k$)"
].median()

score_median = df[
    "Spending Score (1-100)"
].median()


def describe_cluster(row):
    income = row["Average_Income"]
    score = row["Average_Spending_Score"]

    if (
        income >= income_median
        and score >= score_median
    ):
        return "High income / high spending"

    if (
        income >= income_median
        and score < score_median
    ):
        return "High income / low spending"

    if (
        income < income_median
        and score >= score_median
    ):
        return "Low income / high spending"

    return "Low income / low spending"


cluster_summary["Description"] = (
    cluster_summary.apply(
        describe_cluster,
        axis=1
    )
)

description_by_cluster = (
    cluster_summary["Description"].to_dict()
)

df["Cluster Description"] = (
    df["Cluster"].map(
        description_by_cluster
    )
)

print("\nCluster descriptions:")
print(cluster_summary)


# ------------------------------------------------------------
# 12. Save the results
# ------------------------------------------------------------
print(
    "\nSaving analysis results..."
)

customers_file = (
    EXTRAS_DIR /
    "mall_customers_agglomerative.csv"
)

summary_file = (
    EXTRAS_DIR /
    "agglomerative_cluster_summary.csv"
)

evaluation_file = (
    EXTRAS_DIR /
    "agglomerative_model_evaluation.csv"
)

df.to_csv(
    customers_file,
    index=False
)

cluster_summary.to_csv(
    summary_file
)

results.to_csv(
    evaluation_file,
    index=False
)

print("\nFiles saved:")
print(f"- {dendrogram_file}")
print(f"- {silhouette_file}")
print(f"- {clusters_file}")
print(f"- {customers_file}")
print(f"- {summary_file}")
print(f"- {evaluation_file}")

print(
    "\nAgglomerative Clustering analysis "
    "completed successfully."
)
