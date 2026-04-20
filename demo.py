from qsim import Simulator, NoiseModel
from qsim.visualize import plot_histogram, plot_bloch_sphere
import matplotlib.pyplot as plt

# Noise-free
sim = Simulator("input/composer-2026-04-20_16-04.qasm", shots=int(input("how many shots: ")), seed=1)
counts = sim.run()
print(counts)

# Visualize
plot_histogram(counts, title="My circuit")

# With noise
noise = NoiseModel(depolarizing=0.05, amplitude_damping=0.02)
sim = Simulator("input/composer-2026-04-20_16-04.qasm", shots=int(input("how many shots: ")), noise=noise, seed=1)
noisy_counts = sim.run()
plot_histogram(noisy_counts, title="My circuit (noisy)")
sv = sim.statevector()
plot_bloch_sphere(sv, qubit=0, title= "Bloch sphere for qubit ")
plt.show()