"""Tests for the quantum simulator."""
import numpy as np
import pytest

from qsim import Simulator, NoiseModel, parse_qasm
from qsim.sim import apply_single_qubit_gate, apply_two_qubit_gate
from qsim.gates import H, X, CNOT
from qsim.visualize import bloch_vector


# ---------------------------------------------------------------------------
# Gate application primitives
# ---------------------------------------------------------------------------

def test_single_qubit_gate_on_one_qubit():
    # |0> -> H -> (|0> + |1>)/sqrt(2)
    state = np.array([1, 0], dtype=complex)
    state = apply_single_qubit_gate(state, H, 0, 1)
    expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(state, expected, atol=1e-10)


def test_x_gate_flips_bit():
    # |00> apply X to qubit 0 -> |01> (bitstring) which in little-endian is index 1
    state = np.array([1, 0, 0, 0], dtype=complex)
    state = apply_single_qubit_gate(state, X, 0, 2)
    expected = np.array([0, 1, 0, 0], dtype=complex)
    np.testing.assert_allclose(state, expected, atol=1e-10)


def test_x_gate_on_qubit_1():
    # |00> apply X to qubit 1 -> index 2 (binary 10)
    state = np.array([1, 0, 0, 0], dtype=complex)
    state = apply_single_qubit_gate(state, X, 1, 2)
    expected = np.array([0, 0, 1, 0], dtype=complex)
    np.testing.assert_allclose(state, expected, atol=1e-10)


def test_cnot_on_bell_prep():
    # Prepare (|00> + |10>)/sqrt(2), then CNOT(0,1) -> Bell state (|00> + |11>)/sqrt(2)
    state = np.array([1, 0, 0, 0], dtype=complex)
    state = apply_single_qubit_gate(state, H, 0, 2)
    state = apply_two_qubit_gate(state, CNOT, 0, 1, 2)
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(state, expected, atol=1e-10)


def test_cnot_identity_when_control_zero():
    # |01> control=0,target=1 -> CNOT should not flip
    state = np.array([0, 0, 1, 0], dtype=complex)  # index 2 = '10' = qubit 1 is 1
    state = apply_two_qubit_gate(state, CNOT, 0, 1, 2)
    np.testing.assert_allclose(state, np.array([0, 0, 1, 0], dtype=complex), atol=1e-10)


def test_cnot_flips_when_control_one():
    # |01> where qubit 0 = 1, qubit 1 = 0 -> index 1
    state = np.array([0, 1, 0, 0], dtype=complex)
    state = apply_two_qubit_gate(state, CNOT, 0, 1, 2)
    # CNOT flips target (qubit 1) because control (qubit 0) = 1
    # Result: qubit 0 = 1, qubit 1 = 1 -> index 3
    np.testing.assert_allclose(state, np.array([0, 0, 0, 1], dtype=complex), atol=1e-10)


# ---------------------------------------------------------------------------
# QASM parsing
# ---------------------------------------------------------------------------

def test_parse_basic_qasm():
    src = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """
    c = parse_qasm(src)
    assert c.num_qubits == 2
    assert c.num_clbits == 2
    assert len(c.instructions) == 4
    assert c.instructions[0].name == "h"
    assert c.instructions[0].qubits == [0]
    assert c.instructions[1].name == "cx"
    assert c.instructions[1].qubits == [0, 1]
    assert c.instructions[2].name == "measure"


def test_parse_parameterized_gate():
    src = """
    OPENQASM 2.0;
    qreg q[1];
    rx(pi/2) q[0];
    u3(pi/2, 0, pi) q[0];
    """
    c = parse_qasm(src)
    assert len(c.instructions) == 2
    assert c.instructions[0].name == "rx"
    assert c.instructions[0].params == pytest.approx([np.pi / 2])
    assert c.instructions[1].params == pytest.approx([np.pi / 2, 0, np.pi])


# ---------------------------------------------------------------------------
# End-to-end simulation
# ---------------------------------------------------------------------------

def test_bell_state_statevector():
    src = """
    OPENQASM 2.0;
    qreg q[2];
    h q[0];
    cx q[0], q[1];
    """
    sim = Simulator(src, shots=0)
    sv = sim.statevector()
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(sv, expected, atol=1e-10)


def test_bell_state_counts():
    src = """
    OPENQASM 2.0;
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """
    sim = Simulator(src, shots=5000, seed=42)
    counts = sim.run()
    # Should only see '00' and '11', roughly 50/50.
    assert set(counts.keys()) <= {"00", "11"}
    total = sum(counts.values())
    assert total == 5000
    assert abs(counts["00"] / total - 0.5) < 0.03
    assert abs(counts["11"] / total - 0.5) < 0.03


def test_ghz_state_statevector():
    src = """
    OPENQASM 2.0;
    qreg q[3];
    h q[0];
    cx q[0], q[1];
    cx q[1], q[2];
    """
    sim = Simulator(src, shots=0)
    sv = sim.statevector()
    expected = np.zeros(8, dtype=complex)
    expected[0] = 1 / np.sqrt(2)
    expected[7] = 1 / np.sqrt(2)
    np.testing.assert_allclose(sv, expected, atol=1e-10)


def test_single_qubit_rotation():
    # RX(pi) on |0> should give -i|1>
    src = """
    OPENQASM 2.0;
    qreg q[1];
    rx(pi) q[0];
    """
    sim = Simulator(src, shots=0)
    sv = sim.statevector()
    np.testing.assert_allclose(sv, np.array([0, -1j], dtype=complex), atol=1e-10)


def test_bloch_vector_plus_state():
    # |+> = H|0> has Bloch vector (1, 0, 0)
    state = np.array([1, 1], dtype=complex) / np.sqrt(2)
    x, y, z = bloch_vector(state)
    assert x == pytest.approx(1.0, abs=1e-10)
    assert y == pytest.approx(0.0, abs=1e-10)
    assert z == pytest.approx(0.0, abs=1e-10)


def test_bloch_vector_zero_state():
    state = np.array([1, 0], dtype=complex)
    x, y, z = bloch_vector(state)
    assert (x, y, z) == pytest.approx((0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def test_depolarizing_noise_degrades_bell_correlations():
    src = """
    OPENQASM 2.0;
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """
    # With heavy depolarizing noise we should see all four outcomes.
    sim = Simulator(src, shots=4000, noise=NoiseModel(depolarizing=0.3), seed=7)
    counts = sim.run()
    # We should see at least one of the "bad" outcomes.
    bad = counts.get("01", 0) + counts.get("10", 0)
    assert bad > 100


def test_amplitude_damping_on_excited_state():
    # Start in |1>, no other gates, lots of damping -> should often end in |0>.
    src = """
    OPENQASM 2.0;
    qreg q[1];
    creg c[1];
    x q[0];
    measure q[0] -> c[0];
    """
    sim = Simulator(src, shots=2000, noise=NoiseModel(amplitude_damping=0.8), seed=3)
    counts = sim.run()
    # With gamma = 0.8 applied once after the X gate, P(|0>) ~ 0.8.
    p0 = counts.get("0", 0) / 2000
    assert 0.7 < p0 < 0.9