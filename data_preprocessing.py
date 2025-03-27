import pandas as pd

# Load the dataset
file_path = "Coca-Cola_stock_history.csv"
df = pd.read_csv(r'C:\Users\Ridaa\OneDrive\Desktop\ansar\Coca-Cola_stock_history.csv')

# Display dataset info
print(df.info())
print(df.head())
# Handling missing values
df.fillna(method='ffill', inplace=True)

# Feature Engineering: Add Moving Averages & Returns
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['MA_50'] = df['Close'].rolling(window=50).mean()
df['Daily_Return'] = df['Close'].pct_change()

# Drop NA values generated due to rolling calculations
df.dropna(inplace=True)

# Save the cleaned dataset
df.to_csv("cleaned_coca_cola_stock.csv", index=False)
