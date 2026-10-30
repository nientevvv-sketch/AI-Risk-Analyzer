
import openpyxl 

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Paths
DATA_PATH = Path("WTW_Clean_Financial_Model.xlsx")
OUTPUT_DIR = Path("reports/wtw_charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Excel
df = pd.read_excel(DATA_PATH, sheet_name="WTW Financial Model")

# Convert from wide format to a convenient structure
df = df.set_index("Metric")
df.columns = df.columns.astype(str)

years = list(df.columns)

def plot_metric(metrics, title, ylabel, filename):
    """Plot one or more metrics over the available periods."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for metric in metrics:
        values = df.loc[metric, years].astype(float)
        ax.plot(years, values, marker="o", label=metric)

    ax.set_title(title)
    ax.set_xlabel("Reporting period")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


# 1. Revenue
plot_metric(
    ["Revenue"],
    "WTW Revenue Trend",
    "USD billions (as represented in source file)",
    "revenue.png",
)

# 2. Net income and profitability margin
plot_metric(
    ["Net Income", "Profit Margin"],
    "WTW Profitability Metrics",
    "Values use different units — interpret separately",
    "profitability.png",
)

# 3. Current ratio
plot_metric(
    ["Current Ratio"],
    "WTW Current Ratio",
    "Ratio",
    "current_ratio.png",
)

# 4. Free cash flow and debt-to-revenue
plot_metric(
    ["Free Cash Flow", "Debt-to-Revenue"],
    "WTW Cash Flow and Debt Metrics",
    "Values use different units — interpret separately",
    "cashflow_debt.png",
)

print(f"Charts saved to: {OUTPUT_DIR.resolve()}")
print("\nSource data:")
print(df)
