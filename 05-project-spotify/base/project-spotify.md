<div align="center">
  <img src="./logo.png" width="500">
</div>

<br>

# Project: Music Analysis with Supervised and Unsupervised Learning

## Programming Language

This project must be completed entirely in **Python**.

Use Python libraries such as:

- `pandas`;
- `numpy`;
- `matplotlib`;
- `seaborn`;
- `scikit-learn`.

Do not use other programming languages for data preparation, modeling, evaluation, or visualization.

---

## 1. Project Overview

In this project, you will investigate whether the acoustic characteristics of songs can help identify music genres and discover groups of tracks with similar characteristics.

You will use a modified version of **Spotify Tracks Dataset**, available on Kaggle:
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset


The modified dataset contains information such as:

- `danceability`;
- `energy`;
- `valence`;
- `acousticness`;
- `instrumentalness`;
- `speechiness`;
- `liveness`;
- `tempo`;
- `loudness`;
- `duration_ms`;
- `track_genre`.

The project has two main parts:

1. Supervised Learning;
2. Unsupervised Learning.

The central question is:

> Can acoustic features clearly separate music genres, or do different genres show significant overlap in the feature space?

---

# 2. Part 1: Supervised Learning

## 2.1 Research Question

> Can acoustic features classify songs into selected music genres?

In this part, `track_genre` will be used as the target variable.

You will select between 10 and 15 genres and train classification models to predict the genre of each song.

## 2.2 Dataset Preparation

Before training the models, you must:

- inspect the dataset;
- check the number of tracks per genre;
- identify missing values;
- identify duplicate records;
- inspect possible outliers;
- convert Boolean fields into numeric values when necessary;
- document all preprocessing decisions.

## 2.3 Genre Selection

Use between 10 and 15 genres for the supervised learning task.

The selection must follow a rule defined before model evaluation. For example:

- select genres with at least 100 tracks;
- select a fixed group of genres with different musical profiles;
- select the 10 or 15 genres with the largest number of tracks.

You must explain the selection in the report.

Do not select genres because they produce better model results.

Example:

```python
genre_counts = df["track_genre"].value_counts()

min_samples = 100

eligible_genres = (
    genre_counts[genre_counts >= min_samples]
    .index
)

selected_genres = [
    # Add the selected genres here
]

df_model = df[
    df["track_genre"].isin(selected_genres)
].copy()

print("Selected genres:", df_model["track_genre"].nunique())
print(df_model["track_genre"].value_counts())
```

In the report, explain the selection rule. For example:

> We selected 15 genres with at least 100 tracks per genre to create a manageable multiclass classification problem.

## 2.4 Input Features

Use the following features as a starting point:

```python
feature_columns = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature"
]
```

You may test a different feature set as an extension. If you do this, explain the reason and compare the results.

## 2.5 Data Preparation

Convert Boolean values into numeric values when necessary:

```python
df_model["explicit"] = (
    df_model["explicit"]
    .astype(str)
    .str.upper()
    .map({
        "TRUE": 1,
        "FALSE": 0,
        "VERDADEIRO": 1,
        "FALSO": 0
    })
)
```

Check for missing values:

```python
print(df_model[feature_columns].isna().sum())
```

Remove or treat missing values before splitting the data.

Separate the input features and the target:

```python
X = df_model[feature_columns]
y = df_model["track_genre"]
```

## 2.6 Training and Testing

Split the data into training and test sets.

Use a stratified split:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The stratified split keeps approximately the same class proportions in the training and test sets.

Check the class distribution:

```python
print("Training set:")
print(y_train.value_counts())

print("\nTest set:")
print(y_test.value_counts())
```

## 2.7 Models

Train and compare the following models:

1. Logistic Regression;
2. Decision Tree;
3. Linear Support Vector Classifier.

These models provide different perspectives:

- Logistic Regression provides a linear and interpretable baseline;
- Decision Tree uses decision rules and thresholds;
- Linear SVC provides another linear classification approach.

Use `class_weight="balanced"` only when there is meaningful class imbalance.

```python
class_weight = None

# Use "balanced" only when class imbalance is relevant.
```

Create the models with pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("model", DecisionTreeClassifier(
            max_depth=5,
            class_weight=class_weight,
            random_state=42
        ))
    ]),

    "Linear SVC": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearSVC(
            class_weight=class_weight,
            random_state=42
        ))
    ])
}
```

The scaler must be fitted only with the training data. Using a pipeline helps prevent data leakage.

## 2.8 Evaluation Metrics

Evaluate every model with the same test set.

Use the following metrics:

- accuracy;
- balanced accuracy;
- macro precision;
- macro recall;
- macro F1-score;
- weighted F1-score;
- confusion matrix.

The main metric is macro F1-score. It gives the same weight to each genre and prevents larger classes from dominating the evaluation.

```python
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            y_pred
        ),
        "macro_precision": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
    })

    print(f"\n{name}")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

results_df = pd.DataFrame(results)

display(
    results_df.sort_values(
        "macro_f1",
        ascending=False
    )
)
```

Create a confusion matrix for each model.

The analysis must discuss:

- which genres are easier to classify;
- which genres are confused with each other;
- whether the model performs similarly across genres;
- which model obtains the best macro F1-score;
- whether the results differ between training and test data.

## 3. Part 2: Unsupervised Learning

### 3.1 Research Question

Can we discover groups of songs with similar acoustic characteristics without using genre labels?

In this part, `track_genre` must not be used as an input feature during PCA or clustering.

You may use the genre labels after clustering to interpret and evaluate the resulting groups.

### 3.2 Input Features

Use the following features for the clustering analysis:

```python
cluster_features = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature"
]
```

Do not use:

- `track_genre`;
- the selected genres used in supervised learning;
- variables created from `track_genre`;
- predictions from the supervised models;
- any target variable.

You may use all valid tracks or a documented sample. If you use a sample, explain:

- the sample size;
- the sampling method;
- the random seed;
- the reason for using the sample.

### 3.3 Data Preparation

Create a separate dataframe for the unsupervised analysis:

```python
cluster_df = df.dropna(
    subset=cluster_features + ["track_genre"]
).copy()
```

Convert `explicit` into a numeric value when necessary:

```python
cluster_df["explicit"] = (
    cluster_df["explicit"]
    .astype(str)
    .str.upper()
    .map({
        "TRUE": 1,
        "FALSE": 0,
        "VERDADEIRO": 1,
        "FALSO": 0
    })
)
```

Create the input matrix:

```python
X_cluster = cluster_df[cluster_features]
```

Apply standardization:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
```

The features have different units and ranges. Standardization is required before PCA and clustering.

### 3.4 Principal Component Analysis

Use PCA to represent the data in two dimensions:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

print("Total explained variance:")
print(pca.explained_variance_ratio_.sum())
```

Create a visualization:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    alpha=0.25,
    s=8
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Representation of Tracks")
plt.show()
```

Discuss:

- the variance explained by each component;
- the total variance explained;
- the variables associated with each component;
- whether the visualization shows clear groups;
- whether the genres appear visually separated.

PCA is mainly used for visualization. You may apply K-Means to the standardized features or to a larger number of principal components. Explain your choice.

### 3.5 K-Means Clustering

Test several values for the number of clusters.

Use:

- the elbow method;
- silhouette score;
- cluster size;
- visual inspection;
- cluster interpretability.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scores = []

for k in range(2, 16):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    scores.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(
            X_scaled,
            labels
        )
    })

scores_df = pd.DataFrame(scores)

display(scores_df)
```

Create plots for:

- inertia by number of clusters;
- silhouette score by number of clusters.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

axes[0].plot(
    scores_df["k"],
    scores_df["inertia"],
    marker="o"
)

axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters")
axes[0].set_ylabel("Inertia")

axes[1].plot(
    scores_df["k"],
    scores_df["silhouette"],
    marker="o"
)

axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.show()
```

The number of clusters must be selected using:

- quantitative results;
- visualizations;
- cluster sizes;
- cluster profiles;
- interpretability.

Do not use only one criterion.

After selecting a value for `k`, train the final K-Means model:

```python
chosen_k = 10

kmeans = KMeans(
    n_clusters=chosen_k,
    n_init=10,
    random_state=42
)

cluster_df["cluster"] = kmeans.fit_predict(X_scaled)
```

Check the cluster sizes:

```python
cluster_sizes = (
    cluster_df["cluster"]
    .value_counts()
    .sort_index()
)

print("Cluster sizes:")
print(cluster_sizes)
```

### 3.6 Cluster Visualization

Visualize the clusters in PCA space:

```python
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_df["cluster"],
    cmap="tab10",
    alpha=0.35,
    s=8
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters in PCA Space")
plt.colorbar(scatter, label="Cluster")
plt.show()
```

The figure must be accompanied by a written analysis.

Discuss:

- whether the clusters form visible regions;
- whether some clusters overlap;
- whether some clusters are more compact than others;
- whether the PCA visualization represents the full clustering structure;
- whether the selected clusters appear balanced in size.

### 3.7 Cluster Profiles

Calculate the mean feature values for each cluster:

```python
cluster_profile = cluster_df.groupby(
    "cluster"
)[cluster_features].mean()

display(cluster_profile)
```

You may also create a heatmap:

```python
import seaborn as sns

plt.figure(figsize=(14, 7))

sns.heatmap(
    cluster_profile.T,
    cmap="coolwarm",
    annot=False
)

plt.title("Mean Feature Values by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Feature")
plt.show()
```

To make the comparison easier, standardize the cluster profile values:

```python
from scipy.stats import zscore

cluster_profile_z = cluster_profile.apply(
    zscore,
    axis=0
)

plt.figure(figsize=(14, 7))

sns.heatmap(
    cluster_profile_z.T,
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title("Standardized Mean Feature Values by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Feature")
plt.show()
```

Describe each cluster based on the observed feature values.

For example, a cluster may contain tracks with:

- high energy;
- high acousticness;
- low danceability;
- high instrumentalness;
- high speechiness.

Do not assign cluster names before analyzing the data.

The analysis should identify:

- the features with the highest values in each cluster;
- the features with the lowest values in each cluster;
- the differences between clusters;
- whether the clusters represent clear acoustic profiles;
- whether some clusters have similar profiles.

### 3.8 Comparing Clusters with Genres

After creating the clusters, use the genre labels for interpretation:

```python
genre_by_cluster = pd.crosstab(
    cluster_df["cluster"],
    cluster_df["track_genre"],
    normalize="index"
)

display(genre_by_cluster)
```

Analyze:

- the most common genres in each cluster;
- whether a cluster contains many genres;
- whether one genre appears in several clusters;
- whether some genres are more concentrated than others;
- whether the clusters represent acoustic patterns or genre categories.

You may display the most common genres in each cluster:

```python
top_genres_by_cluster = {}

for cluster in sorted(cluster_df["cluster"].unique()):
    cluster_data = cluster_df[
        cluster_df["cluster"] == cluster
    ]

    top_genres = (
        cluster_data["track_genre"]
        .value_counts()
        .head(10)
    )

    top_genres_by_cluster[cluster] = top_genres

    print(f"\nCluster {cluster}")
    print(top_genres)
```

You may also calculate the dominant genre and its proportion in each cluster:

```python
cluster_genre_summary = (
    cluster_df.groupby("cluster")["track_genre"]
    .value_counts(normalize=True)
    .rename("proportion")
    .reset_index()
)

dominant_genres = (
    cluster_genre_summary
    .sort_values(
        ["cluster", "proportion"],
        ascending=[True, False]
    )
    .groupby("cluster")
    .head(1)
)

display(dominant_genres)
```

The clusters do not need to match the declared genres. A result with substantial overlap is valid if it is analyzed correctly.

### 3.9 External Clustering Metrics

You may calculate the following metrics after clustering:

- Adjusted Rand Index;
- Normalized Mutual Information.

```python
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score
)

ari = adjusted_rand_score(
    cluster_df["track_genre"],
    cluster_df["cluster"]
)

nmi = normalized_mutual_info_score(
    cluster_df["track_genre"],
    cluster_df["cluster"]
)

print("Adjusted Rand Index:", ari)
print("Normalized Mutual Information:", nmi)
```

Do not use these metrics to train or select the clusters. Use them only for post-hoc analysis.

Discuss possible reasons for low agreement between clusters and genres:

- different genres can have similar acoustic features;
- one genre can contain several musical styles;
- genre labels can reflect catalog or editorial decisions;
- the dataset does not represent all aspects of music;
- acoustic features alone may not define a genre.

The interpretation must distinguish between acoustic similarity and genre similarity. A low ARI or NMI does not necessarily mean that the clustering failed. It may indicate that the selected audio features do not fully describe the genre labels.

### 3.10 Final Discussion

Summarize the unsupervised learning results.

The discussion should answer:

- whether the data contains visible groups of tracks;
- which value of `k` was selected and why;
- which acoustic features distinguish the clusters;
- whether the clusters are balanced;
- whether the clusters correspond to particular genres;
- whether genres are distributed across several clusters;
- whether the external metrics support the visual and descriptive analysis;
- what limitations affect the interpretation of the clusters.

The final conclusion must be based on the cluster profiles, visualizations, cluster sizes, genre distributions, and external metrics. Do not evaluate the clustering only by its agreement with genre labels.


## 4. Methodological Requirements

### 4.1 Duplicate Records

Check for duplicate records:

```python
print("Exact duplicates:", df.duplicated().sum())
```

If artist and track title fields are available, also check for repeated artist-title combinations:

```python
if {"artists", "track_name"}.issubset(df.columns):
    repeated_artist_tracks = (
        df.groupby(["artists", "track_name"])
        .size()
        .reset_index(name="count")
        .query("count > 1")
        .sort_values("count", ascending=False)
    )

    print(
        "Repeated artist-title combinations:",
        len(repeated_artist_tracks)
    )

    display(repeated_artist_tracks.head(20))
```

If the dataset uses different column names, replace `artists` and `track_name` with the correct names.

Explain whether duplicates were:

- removed;
- retained;
- analyzed separately.

If duplicates are removed, use a documented procedure:

```python
df = df.drop_duplicates().copy()

print("Rows after removing exact duplicates:", len(df))
```

If artist-title combinations are used to identify repeated tracks, consider keeping only one record per combination:

```python
df = df.drop_duplicates(
    subset=["artists", "track_name"]
).copy()
```

Use this procedure only if repeated artist-title combinations represent duplicate records in the dataset.

Duplicates can produce overly optimistic results if identical or very similar tracks appear in both the training and test sets.

The report must explain:

- how duplicates were identified;
- how many records were affected;
- whether the records were removed or retained;
- how the decision affects the supervised and unsupervised analyses.

### 4.2 Class Imbalance

Check the number of examples in every selected genre:

```python
genre_counts = (
    df_model["track_genre"]
    .value_counts()
    .sort_index()
)

print(genre_counts)
```

Calculate the class proportions:

```python
genre_proportions = (
    df_model["track_genre"]
    .value_counts(normalize=True)
    .sort_index()
)

print(genre_proportions)
```

You may visualize the class distribution:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

genre_counts.sort_values().plot(
    kind="barh"
)

plt.xlabel("Number of Tracks")
plt.ylabel("Genre")
plt.title("Number of Tracks per Selected Genre")
plt.tight_layout()
plt.show()
```

Use `class_weight="balanced"` only when there is meaningful class imbalance:

```python
class_weight = "balanced"
```

If the classes have similar sizes, use:

```python
class_weight = None
```

Explain why class weighting is or is not necessary.

The report must discuss:

- the smallest and largest classes;
- the difference between class sizes;
- whether the class distribution is sufficiently balanced;
- the selected value for `class_weight`;
- how class imbalance may affect accuracy and macro F1-score.

Macro F1-score should remain the main metric because it gives the same importance to every genre.

### 4.3 Feature Representation

Discuss the treatment of the following variables:

- `key`;
- `mode`;
- `time_signature`;
- `explicit`.

#### `key`

`key` is represented as a numeric value in the original feature set. This representation may imply an order between musical keys. Explain whether this numeric representation is appropriate for the selected models.

As an extension, compare the numeric representation with a categorical representation:

```python
key_dummies = pd.get_dummies(
    df_model["key"],
    prefix="key",
    dtype=int
)
```

#### `mode`

`mode` is usually represented as a binary variable:

```python
mode_mapping = {
    0: 0,
    1: 1
}

df_model["mode"] = df_model["mode"].map(mode_mapping)
```

Explain the meaning of the values used in the dataset.

#### `time_signature`

`time_signature` is represented numerically in the basic feature set. This may imply that the difference between values is continuous. As an extension, compare this representation with categorical encoding:

```python
time_signature_dummies = pd.get_dummies(
    df_model["time_signature"],
    prefix="time_signature",
    dtype=int
)
```

Explain whether numeric or categorical encoding is more suitable for this variable.

#### `explicit`

Convert Boolean or text values into numeric values when necessary:

```python
df_model["explicit"] = (
    df_model["explicit"]
    .astype(str)
    .str.upper()
    .map({
        "TRUE": 1,
        "FALSE": 0,
        "VERDADEIRO": 1,
        "FALSO": 0
    })
)
```

Check whether the conversion created missing values:

```python
print(
    df_model["explicit"]
    .isna()
    .sum()
)
```

The report must explain:

- which variables were treated as numeric;
- which variables were treated as categorical;
- why each representation was selected;
- whether an alternative representation was tested;
- how the representation affects the model results.

### 4.4 Reproducibility

Use fixed random seeds when possible:

```python
random_state = 42
```

Apply the seed consistently:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

```python
KMeans(
    n_clusters=chosen_k,
    n_init=10,
    random_state=42
)
```

The notebook must include:

- dataset source;
- software and library information;
- data preparation steps;
- complete Python code;
- final results;
- execution instructions.

Record the main software versions:

```python
import sys
import sklearn
import pandas
import numpy
import matplotlib
import seaborn

print("Python:", sys.version)
print("pandas:", pandas.__version__)
print("NumPy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", seaborn.__version__)
```

Document the dataset source:

```python
dataset_source = "Insert the dataset name and source URL here"

print("Dataset source:", dataset_source)
```

Document the main preparation steps:

```python
preparation_steps = [
    "Loaded the dataset",
    "Checked the column names and data types",
    "Checked missing values",
    "Checked duplicate records",
    "Converted Boolean variables when necessary",
    "Selected the input features",
    "Separated input features and target labels",
    "Applied the required preprocessing",
    "Used fixed random seeds"
]

for step in preparation_steps:
    print("-", step)
```

Include execution instructions in the report. For example:

1. Install the required Python libraries.
2. Place the dataset in the expected directory.
3. Update the dataset path if necessary.
4. Confirm that all tables, metrics, and figures are generated.
6. Save the final output files.

All preprocessing decisions, random seeds, software versions, and analysis steps must be documented clearly enough for another person to reproduce the results.


## 5. Deliverables

Each group must submit:

1. an executable Python notebook;
2. an 8- to 12-page report;
3. a 10- to 15-minute presentation;
4. a `README.md` file with execution instructions;
5. the Python code and figures used in the report.

All data preparation, modeling, evaluation, and visualization must be implemented in Python.

### 5.1 Required Files

The final submission should contain at least the following files:

```text
project/
├── README.md
├── report.pdf
├── presentation.pdf
├── requirements.txt
├── data/
├── figures/
└── src/
```

The exact organization may vary, but all required files must be easy to locate.

### 5.2 Suggested Report Structure

The report should contain the following sections:

1. Introduction.
2. Research questions.
3. Dataset description.
4. Data preparation.
5. Genre distribution.
6. Supervised learning.
7. Classifier evaluation.
8. PCA and unsupervised learning.
9. Cluster analysis.
10. Comparison between clusters and genres.
11. Limitations.
12. Conclusions.

The report must include:

- the methods used;
- the main code decisions;
- the relevant tables and figures;
- the evaluation results;
- the interpretation of the results;
- the limitations of the analysis.

### 5.3 README File

The `README.md` file must include:

- project description;
- dataset source;
- required Python version;
- required libraries;
- installation instructions;
- dataset preparation instructions;
- notebook execution instructions;
- expected output files;
- information about the random seed;
- contact information for the group members, when required.

Example structure:

```markdown
# Music Genre Analysis

## Description

This project analyzes music tracks using supervised and unsupervised
machine learning methods.

## Dataset

Dataset source:

- Add the dataset name here.
-

## Requirements

- Python 3.x
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Installation

pip install -r requirements.txt
```

## Reproducibility

The analysis uses `random_state=42` whenever the method supports a
random seed.

## Outputs

The notebook generates:

- supervised learning metrics;
- classification reports;
- confusion matrices;
- PCA visualizations;
- K-Means evaluation plots;
- cluster profiles;
- genre-cluster comparisons.

### 5.4 Python Code and Figures

Submit all Python code used in the analysis.

The submitted code must generate the figures and tables shown in the report.

Figures should include:

- genre distribution;
- confusion matrix for each classifier;
- PCA representation;
- inertia by number of clusters;
- silhouette score by number of clusters;
- K-Means clusters in PCA space;
- cluster profile heatmap;
- genre distribution by cluster, when applicable.

Each figure must have:

- a clear title;
- labeled axes;
- a legend when necessary;
- readable text;
- a reference in the report.

The report must not contain figures or results that cannot be generated from the submitted code.

## 6. Questions to Answer

The report must answer the following questions.

### 6.1 Supervised Learning Questions

#### 1. Are the selected genres balanced?

Use the number of tracks and the class proportions to answer this question.

The analysis should include:

- the number of tracks per genre;
- the smallest and largest class;
- the difference between the classes;
- the effect of class balance on the evaluation.

#### 2. Which classifier obtains the best macro F1-score?

Compare the macro F1-score of:

- Logistic Regression;
- Decision Tree;
- Linear SVC.

Identify the classifier with the highest macro F1-score and explain whether the difference is meaningful.

#### 3. Which genres are easier to classify?

Use the classification report to compare the genre-level metrics.

Discuss:

- precision;
- recall;
- F1-score;
- number of test examples;
- possible acoustic reasons for better performance.

#### 4. Which genres are most frequently confused?

Use the confusion matrices and classification reports.

Identify:

- the most frequent incorrect predictions;
- the pairs of genres that are confused;
- whether the confusion is symmetric;
- possible reasons for the confusion.

### 6.2 PCA and Clustering Questions

#### 5. Do the acoustic features clearly separate the selected genres?

Use the PCA visualization and the explained variance.

Discuss:

- whether visible groups exist;
- whether genres overlap;
- whether the first two components explain enough variance;
- whether the visual result supports clear genre separation.

Genre labels must not be used as input features during PCA or clustering.

#### 6. Which value of `k` was selected for K-Means?

Explain the selected value using more than one criterion:

- elbow method;
- silhouette score;
- cluster size;
- cluster profiles;
- visualization;
- interpretability.

Do not select `k` using only one metric.

#### 7. Are the clusters interpretable?

Use the cluster profiles to answer this question.

Discuss:

- whether each cluster has a distinct feature profile;
- which features have high or low values;
- whether some clusters have similar profiles;
- whether the clusters have useful acoustic interpretations.

#### 8. Do the clusters correspond to the declared genres?

Use the genre-by-cluster table.

Discuss:

- whether clusters contain a dominant genre;
- whether clusters contain multiple genres;
- whether the clusters represent acoustic patterns instead of genre labels;
- whether the external metrics support the interpretation.

#### 9. Does one genre appear in multiple clusters?

Use the cluster and genre distributions.

Identify genres that occur in several clusters and explain possible reasons, such as:

- variation within a genre;
- similar acoustic characteristics across different styles;
- differences between artists or subgenres;
- limitations of the available features.

#### 10. Which features distinguish the clusters?

Use the cluster profile table and heatmap.

Identify:

- features with the largest differences between clusters;
- features with similar values across clusters;
- the features that best describe each cluster;
- whether the differences are large enough to support interpretation.

### 6.3 Limitations and Conclusions

#### 11. Which limitations of the dataset may affect the results?

Discuss relevant limitations, such as:

- duplicate or near-duplicate tracks;
- missing values;
- class imbalance;
- limited genre representation;
- possible errors in genre labels;
- incomplete artist or track metadata;
- limited sample size;
- dependence on one data source;
- acoustic features that do not capture lyrics, culture, history, or production context.

#### 12. Are acoustic features sufficient to define a music genre?

Provide a direct answer based on the results.

Consider:

- classifier performance;
- confusion between genres;
- PCA overlap;
- cluster profiles;
- genre distribution across clusters;
- ARI and NMI values;
- limitations of the dataset.

The conclusion should distinguish between acoustic similarity and complete genre definition.

## 7. Assessment Criteria

| Criterion | Weight |
|---|---:|
| Data preparation and analysis | 15% |
| Exploratory analysis and visualizations | 15% |
| Supervised classification | 20% |
| PCA and clustering | 20% |
| Evaluation and interpretation | 15% |
| Communication quality | 10% |
| Organization and reproducibility | 5% |
| **Total** | **100%** |

### 7.1 Data Preparation and Analysis, 15%

The assessment considers:

- correct data loading;
- missing-value treatment;
- duplicate-record analysis;
- Boolean conversion;
- feature selection;
- target definition;
- correct train-test separation;
- prevention of data leakage.

### 7.2 Exploratory Analysis and Visualizations, 15%

The assessment considers:

- genre distribution analysis;
- appropriate tables and figures;
- readable visualizations;
- correct labels and titles;
- useful interpretation of the exploratory results.

### 7.3 Supervised Classification, 20%

The assessment considers:

- implementation of the required classifiers;
- correct use of pipelines;
- correct feature scaling;
- appropriate train-test splitting;
- use of stratification;
- correct handling of class imbalance;
- complete model comparison.

### 7.4 PCA and Clustering, 20%

The assessment considers:

- correct exclusion of genre labels from the clustering inputs;
- correct standardization;
- appropriate use of PCA;
- testing of multiple values of `k`;
- use of the elbow method and silhouette score;
- analysis of cluster sizes;
- correct implementation of the final K-Means model.

### 7.5 Evaluation and Interpretation, 15%

The assessment considers:

- correct evaluation metrics;
- confusion matrix analysis;
- macro F1-score comparison;
- cluster profile analysis;
- comparison between clusters and genres;
- appropriate use of ARI and NMI;
- discussion of limitations;
- evidence-based conclusions.

### 7.6 Communication Quality, 10%

The assessment considers:

- clear writing;
- logical organization;
- appropriate technical language;
- readable figures and tables;
- effective presentation;
- ability to explain the main results within the required time.

### 7.7 Organization and Reproducibility, 5%

The assessment considers:
- working code;
- Github repository
- inclusion of the code and figures used in the report.
