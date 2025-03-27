import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_coca_cola_stock.csv")

# Plot Closing Price & Moving Averages
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['MA_20'], label='MA 20', linestyle='--')
plt.plot(df['MA_50'], label='MA 50', linestyle='--')

plt.title("Coca-Cola Stock Prices & Moving Averages")
plt.legend()
plt.show()
