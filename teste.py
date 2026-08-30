import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt


x = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
y = [-13.85, 2.12, 5.91, 3.08, 0.15, 2.94, 14.88, 41.21, 88.03, 160.19]

plt.scatter(x, y, label = "Pontos da função")
plt.title("Pontos da função")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()