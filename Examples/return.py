import warnings

# Suppress the specific SSL warning
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
#pip install pandas numpy matplotlib in terminal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress the specific SSL warning
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

# Define the file path
file_path = "/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/msft_data_with_indicators.xlsx"

# Load the Excel file into a DataFrame
print("Loading data...")
df = pd.read_excel(file_path)
print("Data loaded successfully")
print(df.head())  # Display the first few rows to verify it's loaded correctly

# Convert Date column to datetime and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Plot the adjusted close price
plt.figure()
df['Adj Close'].plot()
plt.xlabel("Date")
plt.ylabel("Adjusted")
plt.title("MSFT Price Data")
plt.show(block=False)
print("Plot displayed")

# Calculate daily and monthly returns
print("Calculating daily returns...")
msft_daily_returns = df['Adj Close'].pct_change()
df['Daily Return'] = msft_daily_returns
print("Daily returns calculated")
print(msft_daily_returns.head())

print("Calculating monthly returns...")
msft_monthly_returns = df['Adj Close'].resample('M').ffill().pct_change()
print("Monthly returns calculated")
print(msft_monthly_returns.head())

fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(msft_daily_returns)
ax1.set_xlabel("Date")
ax1.set_ylabel("Percent")
ax1.set_title("MSFT daily returns data")
plt.show()

# Save the updated DataFrame to the Excel file
output_file_path = "/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/msft_data_with_daily_returns.xlsx"
df.to_excel(output_file_path, sheet_name='MSFT Data with Returns')
print(f"Updated data saved to {output_file_path}")

# = plt.figure()
#ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
#ax1.plot(msft_monthly_returns)
#ax1.set_xlabel("Date")
#ax1.set_ylabel("Percent")
#ax1.set_title("MSFT monthly returns data")
#plt.show()

