import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import matplotlib.pyplot as plt

# 1. Specify the file path to your existing Return.xlsx
file_path = '/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/Return.xlsx'

# 2. Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# 3. Ensure the column name "Daily Return" matches sheet's header
column_name = "Daily Return"
if column_name not in df.columns:
    raise ValueError(f"Column '{column_name}' not found in the Excel file.")

# 4. Drop rows that have NaN in the "Daily Return" column
df = df.dropna(subset=[column_name])

# 5. Compute a rank-based percentile
#    - rank(method="first") => ranks 1..N (no ties if unique values)
#    - Then convert rank to a percentile in (0,1) using (rank - 0.5)/N
df["rank"] = df[column_name].rank(method="first")
n = len(df)
df["rank_percentile"] = (df["rank"] - 0.5) / n

# 6. Apply the quantile function (inverse CDF) for the standard normal distribution
df["Daily Return (Transformed)"] = norm.ppf(df["rank_percentile"])

# 7. Split the transformed data into 5 quantiles (Q1..Q5)
#    - Q1 is the lowest 20%, Q5 is the highest 20%
df["Quintile"] = pd.qcut(
    df["Daily Return (Transformed)"], 
    q=5, 
    labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
)

# 8. Save to a new Excel file in the same folder
output_file_path = os.path.join(
    os.path.dirname(file_path), 
    "Return_transformed.xlsx"
)
df.to_excel(output_file_path, index=False)
print("Data transformation completed.")
print(f"New file with transformed column saved to: {output_file_path}")

# 9. Plot a histogram of the transformed data to confirm it's bell-shaped
plt.figure(figsize=(8, 5))
df["Daily Return (Transformed)"].hist(bins=20, edgecolor='black')
plt.title("Histogram of Inverse Normal Transformed Daily Returns")
plt.xlabel("Transformed Value (Z-score)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()
