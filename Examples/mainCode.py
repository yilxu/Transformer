# Import the fetch_data function from msft_data_analysis.py
from functions import fetch_data

# Fetch MSFT data for the specified date range
msft_data = fetch_data(start_date='2013-01-10', end_date='2023-01-10')

# You can now use msft_data as the input to generate technical indicators
# For example, you can calculate some of the technical indicators available in functions.py
from functions import adx, atr, bb, mfi, get_macd as macd, add_stochastic_oscillator

# Example: Calculate some indicators
msft_data = adx(msft_data)
msft_data = atr(msft_data)
msft_data = bb(msft_data)
msft_data = mfi(msft_data)
msft_data = macd(msft_data)
msft_data = add_stochastic_oscillator(msft_data)

# Save the updated data with indicators
msft_data.to_csv('msft_data_with_indicators.csv', index=False)
print("MSFT data with indicators saved to 'msft_data_with_indicators.csv'")
