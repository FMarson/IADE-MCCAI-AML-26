# if you don't have a virtual environment set up, run the following command to create one
#  python -m venv ml
# Activate the virtual environment on Windows: 
#  ml\Scripts\activate
#if it was not allowed to run scripts, you can change the execution policy by running the following command in PowerShell as an administrator:
#  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# pip install scikit-learn matplotlib   

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(
    n_samples=500,
    centers=4,
    random_state=42
)

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

# Create the plot
fig, ax1 = plt.subplots(figsize=(7, 4.5))

# Primary y-axis: inertia
line1 = ax1.plot(
    list(k_range),
    inertias,
    "bo-",
    label="Inertia"
)

ax1.set_xlabel("Number of clusters, k")
ax1.set_ylabel("Inertia", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.set_xticks(list(k_range))

# Secondary y-axis: silhouette score
ax2 = ax1.twinx()

line2 = ax2.plot(
    list(k_range),
    silhouette_scores,
    "go-",
    label="Silhouette score"
)

ax2.set_ylabel("Silhouette score", color="green")
ax2.tick_params(axis="y", labelcolor="green")
ax2.set_ylim(0, 1)

# Combine legends from both axes
lines = line1 + line2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="center right")

plt.title("Elbow Method and Silhouette Score")
fig.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=300, bbox_inches="tight")
plt.show()