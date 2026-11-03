import pandas as pd

df = pd.read_csv(
    "./data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print(df.head())
print()
print(df.shape)
print()
print(df["label"].value_counts())