import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# 1. Create output directories
# ------------------------------------------------------------
FIGURE_DIR = Path("./figures/04a")
EXTRA_DIR = Path("./extras/04a")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
EXTRA_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Load the dataset
# ------------------------------------------------------------
print("\n[1/10] Loading the dataset...")

file_path = Path("./data/mall_customers.csv")

if not file_path.exists():
    raise FileNotFoundError(
        f"Could not find '{file_path}'. "
        "Place the CSV file in the correct folder."
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
print("\n[2/10] Inspecting and cleaning the data...")

print("\nColumn names:")
print(df.columns.tolist())

if "Genre" in df.columns:
    df = df.rename(columns={"Genre": "Gender"})

numeric_columns = [
    "CustomerID",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

required_columns = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

print("\nMissing values before cleaning:")
print(df[required_columns].isna().sum())

df = df.dropna(subset=required_columns).copy()
df = df.reset_index(drop=True)

if len(df) == 0:
    raise ValueError("No usable rows remain after cleaning.")

print(f"\nRows after cleaning: {len(df)}")


# ------------------------------------------------------------
# 4. Select and scale the features
# ------------------------------------------------------------
print("\n[3/10] Selecting and scaling the features...")

feature_columns = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[feature_columns].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features used:")
print(feature_columns)

print(f"\nScaled data shape: {X_scaled.shape}")


# ------------------------------------------------------------
# 5. Estimate suitable eps values
# ------------------------------------------------------------
print("\n[4/10] Estimating DBSCAN eps values...")

min_samples_for_search = 5

nearest_neighbors = NearestNeighbors(
    n_neighbors=min_samples_for_search
)

nearest_neighbors.fit(X_scaled)

distances, indices = nearest_neighbors.kneighbors(X_scaled)

# Distance to the k-th nearest neighbor.
k_distances = np.sort(
    distances[:, min_samples_for_search - 1]
)

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(k_distances) + 1),
    k_distances,
    color="navy",
    linewidth=2
)

plt.title(
    f"k-distance graph, k = {min_samples_for_search}"
)

plt.xlabel("Sorted observations")
plt.ylabel(f"Distance to neighbor {min_samples_for_search}")

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "01_dbscan_k_distance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 6. Compare DBSCAN parameters
# ------------------------------------------------------------
print("\n[5/10] Testing DBSCAN parameters...")

eps_values = np.arange(0.15, 1.01, 0.05)
min_samples_values = [3, 4, 5, 6, 8, 10]

evaluation_rows = []

for min_samples in min_samples_values:
    for eps in eps_values:
        model = DBSCAN(
            eps=float(eps),
            min_samples=int(min_samples),
            metric="euclidean"
        )

        labels = model.fit_predict(X_scaled)

        non_noise_mask = labels != -1
        non_noise_labels = labels[non_noise_mask]

        n_clusters = len(
            set(non_noise_labels)
        )

        n_noise = int(
            np.sum(labels == -1)
        )

        if (
            n_clusters >= 2
            and len(non_noise_labels) > n_clusters
        ):
            score = silhouette_score(
                X_scaled[non_noise_mask],
                non_noise_labels
            )
        else:
            score = np.nan

        evaluation_rows.append({
            "eps": float(eps),
            "min_samples": int(min_samples),
            "n_clusters": n_clusters,
            "noise_points": n_noise,
            "noise_fraction": n_noise / len(labels),
            "Silhouette Score": score
        })

results = pd.DataFrame(evaluation_rows)

valid_results = results.dropna(
    subset=["Silhouette Score"]
).copy()

if valid_results.empty:
    raise ValueError(
        "No DBSCAN configuration produced at least "
        "two non-noise clusters."
    )

# Avoid choosing a solution with almost all points classified as noise.
valid_results = valid_results[
    valid_results["noise_fraction"] <= 0.40
].copy()

if valid_results.empty:
    raise ValueError(
        "All valid DBSCAN configurations classified more "
        "than 40% of the observations as noise."
    )

best_row = valid_results.loc[
    valid_results["Silhouette Score"].idxmax()
]

best_eps = float(best_row["eps"])
best_min_samples = int(best_row["min_samples"])
best_score = float(best_row["Silhouette Score"])

print("\nBest DBSCAN configuration:")
print(best_row.to_string())

print(
    f"\nSelected eps: {best_eps:.2f}"
)

print(
    f"Selected min_samples: {best_min_samples}"
)

print(
    f"Selected silhouette score: {best_score:.4f}"
)

results.to_csv(
    EXTRA_DIR / "dbscan_parameter_evaluation.csv",
    index=False
)


# ------------------------------------------------------------
# 7. Plot evaluation results
# ------------------------------------------------------------
print("\n[6/10] Plotting DBSCAN evaluation results...")

plot_data = valid_results.copy()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=plot_data,
    x="eps",
    y="Silhouette Score",
    hue="min_samples",
    size="n_clusters",
    palette="viridis",
    sizes=(50, 250),
    alpha=0.85
)

plt.scatter(
    best_eps,
    best_score,
    color="red",
    marker="X",
    s=220,
    label="Selected configuration",
    zorder=5
)

plt.title("DBSCAN Parameter Evaluation")
plt.xlabel("eps")
plt.ylabel("Silhouette score")
plt.legend(
    title="min_samples",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "02_dbscan_parameter_evaluation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 8. Train the final DBSCAN model
# ------------------------------------------------------------
print("\n[7/10] Training the final DBSCAN model...")

final_model = DBSCAN(
    eps=best_eps,
    min_samples=best_min_samples,
    metric="euclidean"
)

df["Cluster"] = final_model.fit_predict(X_scaled)

n_clusters = len(
    set(df["Cluster"]) - {-1}
)

n_noise = int(
    (df["Cluster"] == -1).sum()
)

print(f"\nFinal number of clusters: {n_clusters}")
print(f"Noise points: {n_noise}")


# ------------------------------------------------------------
# 9. Visualize the customer segments
# ------------------------------------------------------------
print("\n[8/10] Visualizing customer segments...")

plt.figure(figsize=(10, 6))

plot_df = df.copy()
plot_df["Cluster Label"] = plot_df["Cluster"].astype(str)

sns.scatterplot(
    data=plot_df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster Label",
    palette="tab10",
    s=100,
    alpha=0.85
)

plt.title(
    "Customer Segments Using DBSCAN"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster, -1 = noise")

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "03_dbscan_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 10. Create the cluster summary
# ------------------------------------------------------------
print("\n[9/10] Creating the cluster summary...")

cluster_summary = (
    df.groupby("Cluster")
      .agg(
          Customers=("CustomerID", "count"),
          Average_Age=("Age", "mean"),
          Average_Income=("Annual Income (k$)", "mean"),
          Average_Spending_Score=(
              "Spending Score (1-100)",
              "mean"
          )
      )
      .round(2)
      .sort_index()
)

cluster_summary["Is_Noise"] = (
    cluster_summary.index == -1
)

print("\nCluster summary:")
print(cluster_summary)


# ------------------------------------------------------------
# 11. Add descriptive labels
# ------------------------------------------------------------
print("\n[10/10] Adding cluster descriptions...")

income_median = df["Annual Income (k$)"].median()
score_median = df["Spending Score (1-100)"].median()


def describe_cluster(row):
    if row.name == -1:
        return "Noise / unassigned"

    income = row["Average_Income"]
    score = row["Average_Spending_Score"]

    if income >= income_median and score >= score_median:
        return "High income / high spending"

    if income >= income_median and score < score_median:
        return "High income / low spending"

    if income < income_median and score >= score_median:
        return "Low income / high spending"

    return "Low income / low spending"


cluster_summary["Description"] = cluster_summary.apply(
    describe_cluster,
    axis=1
)

description_by_cluster = cluster_summary[
    "Description"
].to_dict()

df["Cluster Description"] = df["Cluster"].map(
    description_by_cluster
)

print("\nCluster descriptions:")
print(cluster_summary)


# ------------------------------------------------------------
# 12. Evaluate the final model
# ------------------------------------------------------------
non_noise_mask = df["Cluster"] != -1

if (
    n_clusters >= 2
    and non_noise_mask.sum() > n_clusters
):
    final_silhouette = silhouette_score(
        X_scaled[non_noise_mask],
        df.loc[non_noise_mask, "Cluster"]
    )
else:
    final_silhouette = np.nan

print(
    f"\nFinal silhouette score, excluding noise: "
    f"{final_silhouette:.4f}"
)


# ------------------------------------------------------------
# 13. Save the results
# ------------------------------------------------------------
df.to_csv(
    EXTRA_DIR / "mall_customers_dbscan.csv",
    index=False
)

cluster_summary.to_csv(
    EXTRA_DIR / "dbscan_cluster_summary.csv"
)

print("\nFiles saved:")
print(f"- {FIGURE_DIR / '01_dbscan_k_distance.png'}")
print(f"- {FIGURE_DIR / '02_dbscan_parameter_evaluation.png'}")
print(f"- {FIGURE_DIR / '03_dbscan_clusters.png'}")
print(f"- {EXTRA_DIR / 'mall_customers_dbscan.csv'}")
print(f"- {EXTRA_DIR / 'dbscan_cluster_summary.csv'}")
print(f"- {EXTRA_DIR / 'dbscan_parameter_evaluation.csv'}")
