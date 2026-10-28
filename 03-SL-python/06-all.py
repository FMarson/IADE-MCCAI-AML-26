import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


print("Starting program...")


# 1. Load the dataset
print("\n[1/9] Loading the dataset...")

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print(f"Dataset loaded successfully.")
print(f"Number of messages: {len(df)}")
print(f"Number of columns: {len(df.columns)}")


# 2. Convert labels
print("\n[2/9] Converting labels...")

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

if df["label"].isna().any():
    raise ValueError(
        "Unexpected label found in the dataset."
    )

print("Labels converted successfully.")
print("0 = Not spam")
print("1 = Spam")


# 3. Separate inputs and labels
print("\n[3/9] Separating messages and labels...")

X = df["message"]
y = df["label"]

print(f"Number of input messages: {len(X)}")
print(f"Number of labels: {len(y)}")


# 4. Split into training and test data
print("\n[4/9] Splitting data into training and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(f"Training messages: {len(X_train)}")
print(f"Test messages: {len(X_test)}")


# 5. Convert text into numerical TF-IDF features
print("\n[5/9] Converting messages into TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True
)

X_train_vectorized = vectorizer.fit_transform(
    X_train
)

X_test_vectorized = vectorizer.transform(
    X_test
)

print("TF-IDF conversion completed.")
print(
    f"Number of features: "
    f"{X_train_vectorized.shape[1]}"
)


# 6. Create the models
print("\n[6/9] Creating the models...")

models = {
    "Multinomial Naive Bayes": MultinomialNB(),

    "Support Vector Machine": LinearSVC(
        penalty="l2",
        C=1.0,
        random_state=42
    ),

    "k-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",
        algorithm="brute"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        random_state=42
    )
}

print("Models created:")
for model_name in models:
    print(f"- {model_name}")


# 7. Train and evaluate
print("\n[7/9] Training and evaluating models...")

all_results = []

for name, model in models.items():
    print(f"\nTraining model: {name}")

    model.fit(
        X_train_vectorized,
        y_train
    )

    print(f"Making predictions with: {name}")

    predictions = model.predict(
        X_test_vectorized
    )

    print(f"Calculating metrics for: {name}")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
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
            labels=[0, 1],
            target_names=["Not spam", "Spam"],
            zero_division=0
        )
    )

    print(f"Accuracy:    {accuracy:.3f}")
    print(f"Precision:   {precision:.3f}")
    print(f"Recall:      {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 score:    {f1:.3f}")

    print(f"Displaying confusion matrix for: {name}")

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not spam", "Spam"]
    )

    display.plot(cmap="Blues")
    plt.title(name)
    plt.show()
    plt.savefig(f"{name}_confusion_matrix.png")

    print(f"Finished evaluating: {name}")


# 8. Compare results
print("\n[8/9] Comparing model results...")

results_df = pd.DataFrame(all_results)

print("\nModel comparison:")
print(results_df.round(3))


# 9. Classify new messages
print("\n[9/9] Classifying new messages...")

new_messages = [
    "Congratulations! You won a free prize!",
    "Can we meet after class to discuss the project?",
    "Claim your reward now by clicking this link!"
]

print(f"Number of new messages: {len(new_messages)}")

print("Converting new messages into TF-IDF features...")

new_messages_vectorized = vectorizer.transform(
    new_messages
)

print("\nNew message predictions:")

for name, model in models.items():
    print()
    print(name)
    print("-" * len(name))

    new_predictions = model.predict(
        new_messages_vectorized
    )

    for message, prediction in zip(
        new_messages,
        new_predictions
    ):
        label = (
            "Spam"
            if prediction == 1
            else "Not spam"
        )

        print(f"\nMessage: {message}")
        print(f"Prediction: {label}")


print("\nProgram finished successfully.")