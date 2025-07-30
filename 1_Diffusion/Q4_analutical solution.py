import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-0.5,0.5,101)
y = np.cos(3.07343*x)
y /= np.linalg.norm(y)
y_iso = np.cos(3.08350*x)
y_iso /= np.linalg.norm(y_iso)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Anisropic scattering")
plt.plot(x, y_iso, label="Isropic scattering")
plt.xlabel("Position x (m)")
plt.ylabel("Normalized Flux")
plt.legend()
plt.grid()
plt.show()

