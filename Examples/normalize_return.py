import pandas as pd

# Load dataset for MSFT data
file_path = '/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/Return.xlsx'
return_data = pd.read_excel(file_path)

# Proceed with normalization
msft_data_normalized_min_max = (return_data - return_data.min()) / (return_data.max() - return_data.min())

# Save the normalized data to a new Excel file in the same folder
output_file_path = '/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/return_data_normalized_min_max.xlsx'
msft_data_normalized_min_max.to_excel(output_file_path, index=False)

print("Data normalization completed and saved.")

