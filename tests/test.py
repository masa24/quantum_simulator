"""Run all QASM test circuits through the simulator."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import os
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps
from qsim import Simulator
from qsim.visualize import plot_histogram
import matplotlib.pyplot as plt

BASIS_GATES = [
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "id",
    "rx", "ry", "rz", "u1", "u2", "u3",
    "cx", "cz", "swap",
]

# Find all .qasm files in the same folder as this script.
here = os.path.dirname(os.path.abspath(__file__))
qasm_files = sorted(glob.glob(os.path.join(here, "*.qasm")))

for path in qasm_files:
    name = os.path.basename(path)
    print(f"\n=== {name} ===")

    qc = QuantumCircuit.from_qasm_file(path)
    print(f"Original gates:   {dict(qc.count_ops())}")

    qc_basic = transpile(qc, basis_gates=BASIS_GATES, optimization_level=1)
    print(f"Transpiled gates: {dict(qc_basic.count_ops())}")

    sim = Simulator(dumps(qc_basic), shots=1024, seed=1)
    counts = sim.run()
    print(f"Counts: {counts}")

    plot_histogram(counts, title=name)
    plt.show()   # close window to advance to next circuit