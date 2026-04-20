"""Noise channels for the simulator.

Implements two standard channels as stochastic (per-shot) operations
on the pure state vector:

    - Depolarizing: with probability p, apply a uniformly random Pauli.
    - Amplitude damping: stochastic Kraus unraveling using K0/K1
      (normalized on the qubit subspace).

Both are applied after each gate on the affected qubit(s).
"""
import numpy as np
from qsim.gates import X, Y, Z


class NoiseModel:
    """Simple noise model applied per-shot.

    Parameters
    ----------
    depolarizing : float
        Probability (0..1) of a depolarizing error after each single-qubit
        gate. With probability depolarizing, one of X, Y, Z is applied
        uniformly at random.
    amplitude_damping : float
        Decay parameter gamma (0..1) for the amplitude damping channel.
        Simulated stochastically: with probability proportional to gamma
        the qubit relaxes toward |0>.
    """

    def __init__(
        self,
        depolarizing: float = 0.0,
        amplitude_damping: float = 0.0,
    ):
        if not 0.0 <= depolarizing <= 1.0:
            raise ValueError("depolarizing must be in [0, 1]")
        if not 0.0 <= amplitude_damping <= 1.0:
            raise ValueError("amplitude_damping must be in [0, 1]")
        self.depolarizing = depolarizing
        self.amplitude_damping = amplitude_damping

    def is_trivial(self) -> bool:
        return self.depolarizing == 0.0 and self.amplitude_damping == 0.0

    def apply_after_gate(
        self,
        state: np.ndarray,
        qubits: list[int],
        num_qubits: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply noise channels to the affected qubits.

        Called after each gate by the simulator.
        """
        from qsim.sim import apply_single_qubit_gate  # local import: avoid cycle

        for q in qubits:
            # Depolarizing: random Pauli with probability p.
            if self.depolarizing > 0.0 and rng.random() < self.depolarizing:
                pauli = rng.choice([X, Y, Z])
                state = apply_single_qubit_gate(state, pauli, q, num_qubits)

            # Amplitude damping via Kraus unraveling.
            if self.amplitude_damping > 0.0:
                state = _apply_amplitude_damping(
                    state, q, num_qubits, self.amplitude_damping, rng
                )

        return state


def _apply_amplitude_damping(
    state: np.ndarray,
    qubit: int,
    num_qubits: int,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stochastically apply an amplitude damping channel to one qubit.

    Kraus operators:
        K0 = [[1, 0], [0, sqrt(1-gamma)]]   (no decay)
        K1 = [[0, sqrt(gamma)], [0, 0]]     (decay |1> -> |0>)

    For a pure state |psi>, the probability of outcome i is ||K_i |psi>||^2.
    We sample an outcome, apply the corresponding operator, and renormalize.
    """
    # Reshape to isolate the target qubit's axis.
    shape = [2] * num_qubits
    tensor = state.reshape(shape)
    # Move target qubit to axis 0 for convenience.
    tensor = np.moveaxis(tensor, num_qubits - 1 - qubit, 0)
    # tensor[0] = amplitudes where this qubit is |0>
    # tensor[1] = amplitudes where this qubit is |1>

    p1 = np.sum(np.abs(tensor[1]) ** 2)  # prob. qubit is |1>
    # Probability that K1 outcome occurs is gamma * p1.
    if rng.random() < gamma * p1:
        # Decay event: qubit ends up in |0>, old |0> component is killed.
        new0 = tensor[1].copy()  # amplitude transfers
        new1 = np.zeros_like(tensor[1])
        tensor_new = np.stack([new0, new1], axis=0)
    else:
        # No decay: K0 applied. |1> component shrinks by sqrt(1-gamma).
        new0 = tensor[0]
        new1 = np.sqrt(1.0 - gamma) * tensor[1]
        tensor_new = np.stack([new0, new1], axis=0)

    # Renormalize and move axis back.
    norm = np.linalg.norm(tensor_new)
    if norm > 0:
        tensor_new = tensor_new / norm
    tensor_new = np.moveaxis(tensor_new, 0, num_qubits - 1 - qubit)
    return tensor_new.reshape(state.shape)