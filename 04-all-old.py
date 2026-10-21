import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# -----------------------------------
# 1. Load the dataset
# -----------------------------------

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


# -----------------------------------
# 2. Convert labels
# -----------------------------------

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# -----------------------------------
# 3. Separate inputs and labels
# -----------------------------------

X = df["message"]
y = df["label"]


# -----------------------------------
# 4. Split into train and test
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 5. Convert text into TF-IDF vectors
# -----------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# -----------------------------------
# 6. Create the models
# -----------------------------------

models = {
    "Multinomial Naive Bayes": MultinomialNB(),

    "Support Vector Machine": LinearSVC(
        random_state=42
    ),

    "k-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",
        algorithm="brute"
    )
}


# -----------------------------------
# 7. Train and evaluate
# -----------------------------------

all_results = []

for name, model in models.items():
    model.fit(
        X_train_vectorized,
        y_train
    )

    predictions = model.predict(
        X_test_vectorized
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    specificity = tn / (tn + fp)

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    all_results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1
    })

    print()
    print(name)
    print("-" * len(name))
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Not spam", "Spam"],
            zero_division=0
        )
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not spam", "Spam"]
    )

    display.plot(cmap="Blues")
    plt.title(name)
    plt.show()


# -----------------------------------
# 8. Compare results
# -----------------------------------

results_df = pd.DataFrame(all_results)

print(results_df.round(3))
