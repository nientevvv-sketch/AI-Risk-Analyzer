
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data.csv")
TARGET = "Bankrupt?"

df = pd.read_csv(DATA_PATH)

print("=== Dataset overview ===")
print("Shape:", df.shape)
print("First 5 rows:")
print(df.head())

print("\n=== Columns ===")
print("Number of columns:", len(df.columns))
print(df.columns.tolist())

print("\n=== Data types ===")
print(df.dtypes.value_counts())
print(df.dtypes)

print("\n=== Missing values ===")
missing = df.isna().sum().sort_values(ascending=False)
print(missing[missing > 0] if missing.any() else "No missing values")

print("\n=== Duplicate rows ===")
print("Duplicates:", df.duplicated().sum())

print("\n=== Target validation ===")
if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

print("Target values:")
print(df[TARGET].value_counts(dropna=False).sort_index())

print("\nTarget proportions:")
print(df[TARGET].value_counts(normalize=True).sort_index().round(4))

print("\nBankruptcy rate:")
print(f"{df[TARGET].mean():.2%}")

print("\n=== Target checks ===")
print("Missing target values:", df[TARGET].isna().sum())
print("Unexpected target values:",
      sorted(set(df[TARGET].dropna().unique()) - {0, 1}))
