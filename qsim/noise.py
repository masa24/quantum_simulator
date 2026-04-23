import numpy as np
from qsim.gates import X, Y, Z


class NoiseModel:
    

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
        from qsim.sim import apply_single_qubit_gate

        for q in qubits:
            if self.depolarizing > 0.0 and rng.random() < self.depolarizing:
                pauli = rng.choice([X, Y, Z])
                state = apply_single_qubit_gate(state, pauli, q, num_qubits)

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

    shape = [2] * num_qubits
    tensor = state.reshape(shape)
    tensor = np.moveaxis(tensor, num_qubits - 1 - qubit, 0)

    p1 = np.sum(np.abs(tensor[1]) ** 2)
    if rng.random() < gamma * p1:
        new0 = tensor[1].copy()
        new1 = np.zeros_like(tensor[1])
        tensor_new = np.stack([new0, new1], axis=0)
    else:
        new0 = tensor[0]
        new1 = np.sqrt(1.0 - gamma) * tensor[1]
        tensor_new = np.stack([new0, new1], axis=0)

    # Renormalize and move axis back.
    norm = np.linalg.norm(tensor_new)
    if norm > 0:
        tensor_new = tensor_new / norm
    tensor_new = np.moveaxis(tensor_new, 0, num_qubits - 1 - qubit)
    return tensor_new.reshape(state.shape)