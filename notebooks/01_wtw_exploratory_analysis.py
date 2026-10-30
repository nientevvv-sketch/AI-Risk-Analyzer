from pathlib import Path
import pandas as pd

DATA_PATH = Path("Insurance_Brokers_Peer_Ultimate_Model_audit_v4.xlsx")
TARGET = "Company"

# 1. Read the file without headers (as a raw array) to locate the required row.
df_raw = pd.read_excel(DATA_PATH, header=None)

# 2. Find the first row where the word "Company" appears.
header_row_index = None
for idx, row in df_raw.iterrows():
    if row.astype(str).str.contains(TARGET, case=False, na=False).any():
        header_row_index = idx
        break

if header_row_index is None:
    raise ValueError(
        f"Critical error: Column '{TARGET}' was not found in any of the initial rows of the file."
    )

# 3. Reload the file using the correct header row index.
df = pd.read_excel(DATA_PATH, header=header_row_index)

# The rest of the validation code follows the standard workflow.
print("=== Dataset overview ===")
print("Shape:", df.shape)
print("First 5 rows:")
print(df.head())

# ... (rest of the script)
