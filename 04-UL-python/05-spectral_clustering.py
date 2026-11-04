import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# 1. Create output directories
# ------------------------------------------------------------
FIGURE_DIR = Path("./figures/05")
EXTRA_DIR = Path("./extras/05")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
EXTRA_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Load the dataset
# ------------------------------------------------------------
print("\n[1/9] Loading the dataset...")

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
print("\n[2/9] Inspecting and cleaning the data...")

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
print("\n[3/9] Selecting and scaling the features...")

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
# 5. Compare Spectral Clustering parameters
# ------------------------------------------------------------
print("\n[4/9] Testing Spectral Clustering parameters...")

cluster_values = range(2, 11)
neighbor_values = [5, 8, 10, 15, 20]

evaluation_rows = []

for n_neighbors in neighbor_values:
    for n_clusters in cluster_values:

        if n_neighbors >= len(X_scaled):
            continue

        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            n_init=20,
            random_state=42
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels
        )

        evaluation_rows.append({
            "n_clusters": n_clusters,
            "n_neighbors": n_neighbors,
            "Silhouette Score": score
        })

results = pd.DataFrame(evaluation_rows)

best_row = results.loc[
    results["Silhouette Score"].idxmax()
]

best_n_clusters = int(
    best_row["n_clusters"]
)

best_n_neighbors = int(
    best_row["n_neighbors"]
)

best_score = float(
    best_row["Silhouette Score"]
)

print("\nBest Spectral Clustering configuration:")
print(best_row.to_string())

print(
    f"\nSelected number of clusters: "
    f"{best_n_clusters}"
)

print(
    f"Selected number of neighbors: "
    f"{best_n_neighbors}"
)

print(
    f"Selected silhouette score: "
    f"{best_score:.4f}"
)

results.to_csv(
    EXTRA_DIR / "spectral_parameter_evaluation.csv",
    index=False
)


# ------------------------------------------------------------
# 6. Plot the parameter evaluation
# ------------------------------------------------------------
print("\n[5/9] Plotting Spectral Clustering evaluation...")

plt.figure(figsize=(10, 6))

for n_neighbors in neighbor_values:
    subset = results[
        results["n_neighbors"] == n_neighbors
    ]

    plt.plot(
        subset["n_clusters"],
        subset["Silhouette Score"],
        marker="o",
        linewidth=2,
        label=f"n_neighbors = {n_neighbors}"
    )

plt.scatter(
    best_n_clusters,
    best_score,
    color="red",
    marker="X",
    s=220,
    label="Selected configuration",
    zorder=5
)

plt.title(
    "Spectral Clustering: Silhouette Score"
)

plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.xticks(list(cluster_values))
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "01_spectral_parameter_evaluation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 7. Train the final Spectral Clustering model
# ------------------------------------------------------------
print("\n[6/9] Training the final Spectral Clustering model...")

final_model = SpectralClustering(
    n_clusters=best_n_clusters,
    affinity="nearest_neighbors",
    n_neighbors=best_n_neighbors,
    assign_labels="kmeans",
    n_init=20,
    random_state=42
)

df["Cluster"] = final_model.fit_predict(X_scaled)

print(
    f"\nFinal model trained with "
    f"{best_n_clusters} clusters."
)

print(
    f"Graph parameter n_neighbors: "
    f"{best_n_neighbors}"
)


# ------------------------------------------------------------
# 8. Visualize customer segments
# ------------------------------------------------------------
print("\n[7/9] Visualizing customer segments...")

plt.figure(figsize=(10, 6))

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
    "Customer Segments Using Spectral Clustering"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "02_spectral_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 9. Create the cluster summary
# ------------------------------------------------------------
print("\n[8/9] Creating the cluster summary...")

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

print("\nCluster summary:")
print(cluster_summary)


# ------------------------------------------------------------
# 10. Add descriptive labels
# ------------------------------------------------------------
print("\n[9/9] Adding cluster descriptions...")

income_median = df["Annual Income (k$)"].median()
score_median = df["Spending Score (1-100)"].median()


def describe_cluster(row):
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
# 11. Evaluate the final model
# ------------------------------------------------------------
final_silhouette = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print(
    f"\nFinal silhouette score: "
    f"{final_silhouette:.4f}"
)


# ------------------------------------------------------------
# 12. Save the results
# ------------------------------------------------------------
df.to_csv(
    EXTRA_DIR / "mall_customers_spectral.csv",
    index=False
)

cluster_summary.to_csv(
    EXTRA_DIR / "spectral_cluster_summary.csv"
)

print("\nFiles saved:")
print(
    f"- {FIGURE_DIR / '01_spectral_parameter_evaluation.png'}"
)
print(
    f"- {FIGURE_DIR / '02_spectral_clusters.png'}"
)
print(
    f"- {EXTRA_DIR / 'mall_customers_spectral.csv'}"
)
print(
    f"- {EXTRA_DIR / 'spectral_cluster_summary.csv'}"
)
print(
    f"- {EXTRA_DIR / 'spectral_parameter_evaluation.csv'}"
)
