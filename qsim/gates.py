"""Standard quantum gate definitions as NumPy arrays.

All matrices use the computational basis with the convention
|0> = [1, 0], |1> = [0, 1].
"""
import numpy as np

# Single-qubit gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    """Rotation about the X axis by angle theta."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    """Rotation about the Y axis by angle theta."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    """Rotation about the Z axis by angle theta."""
    e_neg = np.exp(-1j * theta / 2)
    e_pos = np.exp(1j * theta / 2)
    return np.array([[e_neg, 0], [0, e_pos]], dtype=complex)


def u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """General single-qubit rotation (OpenQASM u3)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=complex,
    )


def u2(phi: float, lam: float) -> np.ndarray:
    """u2 gate: u3(pi/2, phi, lam)."""
    return u3(np.pi / 2, phi, lam)


def u1(lam: float) -> np.ndarray:
    """u1 gate: phase gate diag(1, e^{i lam})."""
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=complex)


# Two-qubit gates (in the ordering |c t>, control first)
CNOT = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=complex,
)

CZ = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ],
    dtype=complex,
)

SWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=complex,
)


SINGLE_QUBIT_GATES = {
    "id": I,
    "i": I,
    "x": X,
    "y": Y,
    "z": Z,
    "h": H,
    "s": S,
    "sdg": SDG,
    "t": T,
    "tdg": TDG,
}

PARAMETERIZED_GATES = {
    "rx": rx,
    "ry": ry,
    "rz": rz,
    "u1": u1,
    "u2": u2,
    "u3": u3,
    "u": u3,  # OpenQASM 3 alias
    "p": u1,  # phase gate alias
}

TWO_QUBIT_GATES = {
    "cx": CNOT,
    "cnot": CNOT,
    "cz": CZ,
    "swap": SWAP,
}