import os
import pandas as pd
import numpy as np

def scale_df_to_discrete_5(df):
    """
    Scale each column in df to the discrete set {1,2,3,4,5} using min-max scaling
    and rounding to the nearest integer.
    """
    new_min, new_max = 1, 5  # Target range [1,5]

    # Calculate min and max for each column
    col_min = df.min(axis=0) 
    col_max = df.max(axis=0) 

    # Avoid division by zero for columns with constant values
    col_range = col_max - col_min
    col_range.replace(0, 1e-10, inplace=True)  # Small non-zero value

    # Apply min-max scaling
    df_scaled = ((df - col_min) / col_range) * (new_max - new_min) + new_min

    # Round values to the nearest integer (1, 2, 3, 4, or 5)
    df_scaled = df_scaled.round().astype(int)

    # Ensure values remain in the range [1,5] (clip any rounding errors)
    df_scaled = df_scaled.clip(new_min, new_max)

    return df_scaled

if __name__ == "__main__":
    # Path to your original Excel file
    excel_path = (
        "/Users/yilxu/Desktop/Document/PhD/Research/"
        "Hemang Subramanian/Technical/return_data_normalized_min_max.xlsx"
    )
    
    # Read into DataFrame (assuming all numeric columns)
    df = pd.read_excel(excel_path)

    # Scale each column to {1,2,3,4,5}
    df_scaled = scale_df_to_discrete_5(df)

    # Save as a new Excel file in the same folder
    new_excel_path = os.path.join(
        os.path.dirname(excel_path),
        "return_data_5.xlsx"
    )
    df_scaled.to_excel(new_excel_path, index=False)

    print(f"Scaled DataFrame saved to: {new_excel_path}")
