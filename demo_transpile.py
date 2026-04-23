from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps
from qsim import Simulator, NoiseModel
from qsim.visualize import plot_histogram, plot_bloch_sphere
import matplotlib.pyplot as plt

BASIS_GATES = [
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "id",
    "rx", "ry", "rz", "u1", "u2", "u3",
    "cx", "cz", "swap",
]

# Load & transpile
qc = QuantumCircuit.from_qasm_file("input/composer-2026-04-20_16-04.qasm")
print("Original gates:", dict(qc.count_ops()))

qc_basic = transpile(qc, basis_gates=BASIS_GATES, optimization_level=1)
print("Transpiled gates:", dict(qc_basic.count_ops()))

# run simulator
qasm_text = dumps(qc_basic)
print(qasm_text)
sim = Simulator(qasm_text, shots=1024, seed=1)
counts = sim.run()
print("Counts:", counts)

plot_histogram(counts, title="My circuit (noise-free)")

noise = NoiseModel(depolarizing=0.02, amplitude_damping=0.01)
sim = Simulator(qasm_text, shots=1024, noise=noise, seed=1)
noisy_counts = sim.run()
print("Noisy counts:", noisy_counts)

plot_histogram(noisy_counts, title="My circuit (noisy)")


sv = sim.statevector()
plot_bloch_sphere(sv, title="Final state on Bloch sphere")

plt.show()


