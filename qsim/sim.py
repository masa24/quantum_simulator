"""Core state-vector simulator.

State convention: little-endian (qubit 0 is the least significant bit).
This matches OpenQASM / Qiskit so bitstrings like '011' mean
qubit 2 = 0, qubit 1 = 1, qubit 0 = 1.

Gates are applied by reshaping the 2^n state vector into an n-axis
tensor and contracting against the gate matrix on the relevant axes.
This runs in O(2^n) per gate rather than O(4^n) for the naive approach.
"""
from __future__ import annotations

import numpy as np

from qsim.gates import (
    PARAMETERIZED_GATES,
    SINGLE_QUBIT_GATES,
    TWO_QUBIT_GATES,
)
from qsim.noise import NoiseModel
from qsim.parser import Circuit, Instruction, parse_qasm, parse_qasm_file


# -----------------------------------------------------------------------------
# Gate application via tensor contraction
# -----------------------------------------------------------------------------

def apply_single_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    qubit: int,
    num_qubits: int,
) -> np.ndarray:
    """Apply a 2x2 gate to `qubit` in a state vector of length 2^n.

    The state is reshaped to (2, 2, ..., 2) with axis ordering such that
    axis 0 corresponds to the most-significant qubit (qubit n-1) and axis
    n-1 corresponds to qubit 0. This is the standard little-endian layout
    when unraveling a flat array with np.reshape.
    """
    shape = [2] * num_qubits
    tensor = state.reshape(shape)
    # Axis index for `qubit` in little-endian layout.
    axis = num_qubits - 1 - qubit
    # Contract gate[new, old] with tensor along `axis`.
    # tensordot puts the contracted axis at the end, so move it back.
    new_tensor = np.tensordot(gate, tensor, axes=([1], [axis]))
    new_tensor = np.moveaxis(new_tensor, 0, axis)
    return new_tensor.reshape(state.shape)


def apply_two_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    control: int,
    target: int,
    num_qubits: int,
) -> np.ndarray:
    """Apply a 4x4 gate to (control, target).

    The 4x4 gate is interpreted in the basis |c t> where c is the control.
    We reshape to a (2,2,2,2)-on-two-axes form and contract.
    """
    if control == target:
        raise ValueError("control and target must differ")

    shape = [2] * num_qubits
    tensor = state.reshape(shape)

    c_axis = num_qubits - 1 - control
    t_axis = num_qubits - 1 - target

    # Reshape gate to rank-4: [c_out, t_out, c_in, t_in].
    g = gate.reshape(2, 2, 2, 2)

    # Contract g's input axes (2, 3) with tensor's (c_axis, t_axis).
    new_tensor = np.tensordot(g, tensor, axes=([2, 3], [c_axis, t_axis]))
    # After tensordot, the first two axes of new_tensor are [c_out, t_out]
    # and the remaining axes are the original axes with c_axis and t_axis removed.
    # We need to put c_out back at c_axis and t_out back at t_axis.

    # Figure out where the remaining axes currently live.
    remaining_axes = [i for i in range(num_qubits) if i != c_axis and i != t_axis]
    # new_tensor axes order: [c_out, t_out, *remaining_axes_in_original_order]
    # Build the permutation that sends this to original ordering with c_out
    # at c_axis and t_out at t_axis.
    perm = [0] * num_qubits
    # The "source" axes in new_tensor: 0 -> c_axis, 1 -> t_axis,
    # and 2..n+1 -> remaining_axes in their original order.
    source_to_dest = {0: c_axis, 1: t_axis}
    for i, orig_ax in enumerate(remaining_axes):
        source_to_dest[2 + i] = orig_ax
    # perm[dest] = source
    for src, dest in source_to_dest.items():
        perm[dest] = src

    new_tensor = np.transpose(new_tensor, perm)
    return new_tensor.reshape(state.shape)


# -----------------------------------------------------------------------------
# Gate resolution
# -----------------------------------------------------------------------------

def _resolve_gate(instr: Instruction) -> tuple[np.ndarray, str]:
    """Return (matrix, kind) for a gate instruction.

    kind is 'single' or 'two'.
    """
    name = instr.name.lower()

    if name in SINGLE_QUBIT_GATES:
        return SINGLE_QUBIT_GATES[name], "single"
    if name in PARAMETERIZED_GATES:
        return PARAMETERIZED_GATES[name](*instr.params), "single"
    if name in TWO_QUBIT_GATES:
        return TWO_QUBIT_GATES[name], "two"

    raise NotImplementedError(f"Unknown gate: {name!r}")


# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------

class Simulator:
    """State-vector quantum circuit simulator.

    Parameters
    ----------
    source : str or Circuit
        Either QASM source code, a path to a .qasm file, or a pre-parsed Circuit.
    shots : int
        Number of measurement samples. If 0, run noise-free and return only
        the final state vector (no measurements).
    noise : NoiseModel, optional
        Noise model applied after each gate. If None, simulate noise-free.
    seed : int, optional
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        source: str | Circuit,
        shots: int = 1024,
        noise: NoiseModel | None = None,
        seed: int | None = None,
    ):
        if isinstance(source, Circuit):
            self.circuit = source
        elif isinstance(source, str):
            # Heuristic: looks like a file path?
            if source.endswith(".qasm") or "\n" not in source and len(source) < 256:
                try:
                    self.circuit = parse_qasm_file(source)
                except (FileNotFoundError, OSError):
                    self.circuit = parse_qasm(source)
            else:
                self.circuit = parse_qasm(source)
        else:
            raise TypeError("source must be a Circuit, QASM string, or file path")

        self.shots = int(input("how many shots: "))
        self.noise = noise or NoiseModel()
        self.rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def statevector(self) -> np.ndarray:
        """Return the final state vector with no noise and no measurement.

        Measurement instructions are ignored.
        """
        n = self.circuit.num_qubits
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0

        for instr in self.circuit.instructions:
            if instr.name == "measure":
                continue
            state = self._apply_instruction(state, instr, apply_noise=False)

        return state

    def run(self) -> dict[str, int]:
        """Run the circuit `shots` times and return measurement counts.

        Returns
        -------
        counts : dict
            Mapping from bitstring (little-endian, qubit 0 on the right) to
            number of occurrences.
        """
        n = self.circuit.num_qubits
        counts: dict[str, int] = {}

        # If the circuit has no noise and no mid-circuit measurement, we can
        # compute the final distribution once and sample from it.
        if self.noise.is_trivial():
            state = self.statevector()
            # Identify measured qubits in circuit order.
            measured_qubits = [
                i.qubits[0] for i in self.circuit.instructions if i.name == "measure"
            ]
            if not measured_qubits:
                # No measurements: nothing to sample. Return empty counts.
                return {}
            probs = np.abs(state) ** 2
            probs = probs / probs.sum()  # numerical safety
            outcomes = self.rng.choice(2**n, size=self.shots, p=probs)
            for outcome in outcomes:
                bitstring = self._format_outcome(outcome, measured_qubits, n)
                counts[bitstring] = counts.get(bitstring, 0) + 1
            return counts

        # Noisy path: re-simulate each shot.
        for _ in range(self.shots):
            state = np.zeros(2**n, dtype=complex)
            state[0] = 1.0
            measured_qubits = []
            for instr in self.circuit.instructions:
                if instr.name == "measure":
                    measured_qubits.append(instr.qubits[0])
                    continue
                state = self._apply_instruction(state, instr, apply_noise=True)
            if not measured_qubits:
                continue
            probs = np.abs(state) ** 2
            probs = probs / probs.sum()
            outcome = self.rng.choice(2**n, p=probs)
            bitstring = self._format_outcome(outcome, measured_qubits, n)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _apply_instruction(
        self, state: np.ndarray, instr: Instruction, apply_noise: bool
    ) -> np.ndarray:
        n = self.circuit.num_qubits
        gate, kind = _resolve_gate(instr)
        if kind == "single":
            state = apply_single_qubit_gate(state, gate, instr.qubits[0], n)
        elif kind == "two":
            state = apply_two_qubit_gate(
                state, gate, instr.qubits[0], instr.qubits[1], n
            )
        else:
            raise NotImplementedError(kind)

        if apply_noise and not self.noise.is_trivial():
            state = self.noise.apply_after_gate(state, instr.qubits, n, self.rng)

        return state

    @staticmethod
    def _format_outcome(outcome: int, measured_qubits: list[int], n: int) -> str:
        """Format a measurement outcome as a bitstring.

        Only the measured qubits appear, in the order they were measured,
        printed right-to-left (qubit measured last appears rightmost, matching
        the typical QASM convention).
        """
        # Full n-bit representation.
        full = format(outcome, f"0{n}b")  # MSB first, big-endian string
        # full[-1 - q] is the bit for qubit q (little-endian).
        bits = [full[-1 - q] for q in measured_qubits]
        # Conventional display: qubit measured first on the left.
        return "".join(bits)