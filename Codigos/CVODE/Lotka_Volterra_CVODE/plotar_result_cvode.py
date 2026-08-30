import pandas as pd
import matplotlib.pyplot as plt
 
# Ler o arquivo CSV
df = pd.read_csv("lotka_volterra_results.csv")

# Plotar os dados
plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['Prey'], label='Prey')
plt.plot(df['Time'], df['Predator'], label='Predator')
plt.title('Lotka-Volterra Model')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()
