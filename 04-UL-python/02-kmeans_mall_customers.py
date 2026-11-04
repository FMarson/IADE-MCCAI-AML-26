# Required packages:
# python -m pip install -r requirements.txt

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

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# 1. Define project directories
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures" / "02"
EXTRAS_DIR = BASE_DIR / "extras" / "02"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
EXTRAS_DIR.mkdir(parents=True, exist_ok=True)


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
    df = df.rename(columns={"Genre": "Gender"})

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

missing_values = df[required_columns].isna().sum()

print("\nMissing values:")
print(missing_values)

df = df.dropna(
    subset=required_columns
).copy()

df = df.reset_index(drop=True)

print(f"\nRows after cleaning: {len(df)}")

if len(df) == 0:
    raise ValueError(
        "The dataset contains no usable rows after cleaning."
    )


# ------------------------------------------------------------
# 4. Explore the data
# ------------------------------------------------------------
print("\n[3/9] Exploring the data...")

print("\nDescriptive statistics:")
print(df.describe())

print("\nGender distribution:")
print(df["Gender"].value_counts())

print("\nBasic visual exploration...")

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

sns.histplot(
    data=df,
    x="Annual Income (k$)",
    bins=15,
    kde=True,
    ax=axes[0],
    color="steelblue"
)

axes[0].set_title(
    "Annual Income Distribution"
)

axes[0].set_xlabel(
    "Annual Income (k$)"
)

sns.histplot(
    data=df,
    x="Spending Score (1-100)",
    bins=15,
    kde=True,
    ax=axes[1],
    color="darkorange"
)

axes[1].set_title(
    "Spending Score Distribution"
)

axes[1].set_xlabel(
    "Spending Score (1-100)"
)

plt.tight_layout()

exploration_figure = (
    FIGURE_DIR / "00_data_exploration.png"
)

plt.savefig(
    exploration_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 5. Select the features for clustering
# ------------------------------------------------------------
print("\n[4/9] Selecting features for clustering...")

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
# 6. Scale the features
# ------------------------------------------------------------
print("\n[5/9] Scaling the features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Features scaled successfully.")
print(f"Scaled data shape: {X_scaled.shape}")


# ------------------------------------------------------------
# 7. Compare possible numbers of clusters
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

    inertias.append(
        model.inertia_
    )

    silhouette_scores.append(
        silhouette_score(
            X_scaled,
            labels
        )
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

best_silhouette = results[
    "Silhouette Score"
].max()

print(
    f"\nBest k according to the silhouette score: "
    f"{best_k}"
)

print(
    f"Best silhouette score: "
    f"{best_silhouette:.4f}"
)


# ------------------------------------------------------------
# 8. Plot the elbow and silhouette results
# ------------------------------------------------------------
print("\n[7/9] Plotting model evaluation results...")

fig, ax1 = plt.subplots(
    figsize=(9, 5)
)

# Inertia axis
line1 = ax1.plot(
    list(k_values),
    inertias,
    marker="o",
    color="blue",
    linewidth=2,
    label="Inertia"
)

ax1.set_xlabel(
    "Number of clusters, k"
)

ax1.set_ylabel(
    "Inertia",
    color="blue"
)

ax1.tick_params(
    axis="y",
    labelcolor="blue"
)

ax1.set_xticks(
    list(k_values)
)

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

ax2.set_ylabel(
    "Silhouette score",
    color="green"
)

ax2.tick_params(
    axis="y",
    labelcolor="green"
)

ax2.set_ylim(
    0,
    1
)

# Highlight the best silhouette score
ax2.scatter(
    best_k,
    best_silhouette,
    color="red",
    s=100,
    zorder=5
)

# Combine legends from both axes
lines = line1 + line2

labels = [
    line.get_label()
    for line in lines
]

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

plt.title(
    "Choosing the Number of Clusters"
)

fig.tight_layout()

evaluation_figure = (
    FIGURE_DIR / "01_model_evaluation.png"
)

plt.savefig(
    evaluation_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 9. Train and visualize the final K-Means model
# ------------------------------------------------------------
print(
    "\n[8/9] Training and visualizing "
    "the final K-Means model..."
)

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = final_kmeans.fit_predict(
    X_scaled
)

# Convert centers from standardized coordinates
# back to the original feature units.
cluster_centers = scaler.inverse_transform(
    final_kmeans.cluster_centers_
)

centers_df = pd.DataFrame(
    cluster_centers,
    columns=feature_columns
)

centers_df.index.name = "Cluster"

print(
    f"\nFinal model trained with k = {best_k}"
)

print(
    "\nCluster centers in original units:"
)

print(
    centers_df.round(2)
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

plt.title(
    f"Customer Clusters Using K-Means, k = {best_k}"
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

clusters_figure = (
    FIGURE_DIR / "02_customer_clusters.png"
)

plt.savefig(
    clusters_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 10.
