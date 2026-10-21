import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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


# 1. Load the dataset
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


# 2. Convert labels into numbers
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# 3. Separate input messages and labels
X = df["message"]
y = df["label"]


# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# 5. Convert text into numerical TF-IDF (Term Frequency - Inverse Document Frequency) features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# 6. Create and train the model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_vectorized,
    y_train
)


# 7. Make predictions
y_pred = model.predict(X_test_vectorized)


# 8. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not spam", "Spam"]
)

display.plot(cmap="Blues")
plt.title("Logistic Regression")
plt.show()


# 9. Calculate metrics
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

specificity = tn / (tn + fp)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\nMetrics:")
print(f"Accuracy:    {accuracy:.3f}")
print(f"Precision:   {precision:.3f}")
print(f"Recall:      {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"F1 score:    {f1:.3f}")


# 10. Detailed classification report
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not spam", "Spam"],
        zero_division=0
    )
)


# 11. Classify new messages
new_messages = [
    "Congratulations! You won a free prize!",
    "Can we meet after class to discuss the project?",
    "Claim your reward now by clicking this link!"
]

new_messages_vectorized = vectorizer.transform(new_messages)

new_predictions = model.predict(
    new_messages_vectorized
)

print("\nNew message predictions:")

for message, prediction in zip(
    new_messages,
    new_predictions
):
    label = "Spam" if prediction == 1 else "Not spam"

    print(f"\nMessage: {message}")
    print(f"Prediction: {label}")