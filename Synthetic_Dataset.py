import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 1000
sigma = 0.1

angles = [180, 135, 90, 45, 0, -45, -90, -135]
radians = np.radians(angles)

X_ideal = 3 * np.cos(radians)
Y_ideal = 1.5 * np.sin(radians)

data = []

for _ in range(n):
    noise_x = np.random.normal(0, sigma, 8)
    noise_y = np.random.normal(0, sigma, 8)

    points = []

    for i in range(8):
        points.append(X_ideal[i] + noise_x[i])
        points.append(Y_ideal[i] + noise_y[i])

    data.append(points)

columns = [f'{c}{i}' for i in range(1, 9) for c in 'xy']
df = pd.DataFrame(data, columns=columns)

fig, axes = plt.subplots(1, 5, figsize=(12, 3))

for k in range(5):
    row = df.iloc[k]

    x = [row[f'x{i}'] for i in range(1, 9)]
    y = [row[f'y{i}'] for i in range(1, 9)]

    x.append(x[0])
    y.append(y[0])

    axes[k].plot(x, y, 'o-', markersize=4)
    axes[k].set_aspect('equal')
    axes[k].grid(True, alpha=0.3)
    axes[k].set_title(f'Shape {k+1}')
    axes[k].set_xlim(-4, 4)

plt.tight_layout()
plt.show()

print(df.head())