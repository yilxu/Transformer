import os
import pandas as pd
import numpy as np
import math

def scale_df_to_5(df):
    """
    #convert 1-5 and check histogram to see distribution
    Scale each column in df to [1,5] by:
        scaled_value = ceil( 255 * (value / max_of_that_column) ).
    Any columns with zero max become all zeros to avoid division by zero.
    """
    # 1) Calculate the max for each column
    col_max = df.max(axis=0)  # Series of maxima, one per column
    
    # 2) For any max==0, replace with a small non-zero to avoid /0
    col_max = col_max.replace(0, 1e-10)
    
    df_scaled = df.div(col_max, axis=1).mul(5)
    df_scaled = np.ceil(df_scaled).clip(0, 5)  
    
    # 4) Convert to unsigned int (so e.g. 123.0 -> 123)
    df_scaled = df_scaled.astype(np.uint8)
    
    return df_scaled

if __name__ == "__main__":
    # 1) Path to your original Excel file
    excel_path = (
        "/Users/yilxu/Desktop/Document/PhD/Research/"
        "Hemang Subramanian/Technical/return_data_normalized_min_max.xlsx"
    )
    
    # 2) Read into DataFrame (assuming all numeric columns)
    df = pd.read_excel(excel_path)
    
    # 3) Scale each column to [0..5] with ceiling
    df_scaled = scale_df_to_5(df)
    
    # 4) Save as a new Excel file in the same folder
    new_excel_path = os.path.join(
        os.path.dirname(excel_path),
        "return_data_5.xlsx"
    )
    df_scaled.to_excel(new_excel_path, index=False)
    
    print(f"Scaled DataFrame saved to: {new_excel_path}")

