"""Convert any IBM Quantum Experience QASM file into a QASM file
that uses only gates our simulator supports."""

from qiskit import QuantumCircuit, transpile
import sys

# Gates our simulator can handle.
BASIS_GATES = [
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "id",
    "rx", "ry", "rz", "u1", "u2", "u3",
    "cx", "cz", "swap",
]


def convert(input_path: str, output_path: str) -> None:
    """Read a QASM file, transpile it to our basis gates, and save it."""
    qc = QuantumCircuit.from_qasm_file(input_path)
    print(f"Original circuit: {qc.count_ops()}")

    qc_basic = transpile(qc, basis_gates=BASIS_GATES, optimization_level=1)
    print(f"Transpiled circuit: {qc_basic.count_ops()}")

    qasm_text = qc_basic.qasm()
    with open(output_path, "w") as f:
        f.write(qasm_text)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python transpile_circuit.py <input.qasm> <output.qasm>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])