from pathlib import Path
import pandas as pd

DATA_PATH = Path("Insurance_Brokers_Peer_Ultimate_Model_audit_v4.xlsx")
TARGET = "Company"

# 1. Читаем файл БЕЗ заголовков (как чистый массив данных), чтобы найти нужную строку
df_raw = pd.read_excel(DATA_PATH, header=None)

# 2. Ищем, в какой строке впервые встречается слово "Company"
header_row_index = None
for idx, row in df_raw.iterrows():
    if row.astype(str).str.contains(TARGET, case=False, na=False).any():
        header_row_index = idx
        break

if header_row_index is None:
    raise ValueError(f"Критическая ошибка: Колонка '{TARGET}' не найдена ни в одной из первых строк файла!")

# 3. Перечитываем файл с правильным индексом строки заголовков
df = pd.read_excel(DATA_PATH, header=header_row_index)

# === Дальше ваш стандартный код проверки ===
print("=== Dataset overview ===")
print("Shape:", df.shape)
print("First 5 rows:")
print(df.head())

# ... (остальная часть вашего скрипта)
