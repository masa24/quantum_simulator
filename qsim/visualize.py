from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


def plot_histogram(
    counts: dict[str, int],
    title: str = "Measurement outcomes",
    ax=None,
    sort: bool = True,
):
    """Bar chart of measurement counts.

    Parameters
    ----------
    counts : dict
        Mapping bitstring -> number of shots.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Axis to draw on. If None, a new figure is created.
    sort : bool
        If True, sort bitstrings lexicographically.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    items = list(counts.items())
    if sort:
        items.sort(key=lambda x: x[0])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    total = sum(values) if values else 1

    bars = ax.bar(labels, values, color="#4c72b0", edgecolor="black")
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v}\n({v / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("Bitstring")
    ax.set_ylabel("Counts")
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    return ax


def _reduced_density_matrix(state: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    """Trace out all qubits except `qubit`, returning a 2x2 density matrix."""
    shape = [2] * num_qubits
    tensor = state.reshape(shape)
    # Move the target qubit's axis to position 0 in the little-endian layout.
    axis = num_qubits - 1 - qubit
    tensor = np.moveaxis(tensor, axis, 0)
    # Flatten the remaining axes.
    flat = tensor.reshape(2, -1)
    # rho = flat @ flat.conj().T
    return flat @ flat.conj().T


def bloch_vector(state: np.ndarray, qubit: int = 0, num_qubits: int | None = None):
    """Compute the Bloch vector (x, y, z) for a qubit.

    Works for pure or mixed single-qubit reduced states.
    """
    if num_qubits is None:
        num_qubits = int(np.log2(len(state)))
    rho = _reduced_density_matrix(state, qubit, num_qubits)
    x = 2.0 * np.real(rho[0, 1])
    y = 2.0 * np.imag(rho[1, 0])
    z = np.real(rho[0, 0] - rho[1, 1])
    return float(x), float(y), float(z)


def plot_bloch_sphere(
    state: np.ndarray,
    qubit: int = 0,
    title: str | None = None,
    ax=None,
):
    """Plot a single qubit's reduced state on the Bloch sphere."""
    num_qubits = int(np.log2(len(state)))
    x, y, z = bloch_vector(state, qubit, num_qubits)

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")

    # Wireframe sphere.
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    sx = np.cos(u) * np.sin(v)
    sy = np.sin(u) * np.sin(v)
    sz = np.cos(v)
    ax.plot_wireframe(sx, sy, sz, color="lightgray", linewidth=0.3, alpha=0.6)

    # Axes.
    for axis_vec, label in [
        ((1, 0, 0), "x"),
        ((0, 1, 0), "y"),
        ((0, 0, 1), "|0>"),
        ((0, 0, -1), "|1>"),
    ]:
        ax.plot([0, axis_vec[0]], [0, axis_vec[1]], [0, axis_vec[2]], color="gray", linewidth=0.5)
        ax.text(*[1.1 * c for c in axis_vec], label, fontsize=9)

    # State vector as an arrow from origin.
    ax.quiver(0, 0, 0, x, y, z, color="crimson", linewidth=2, arrow_length_ratio=0.1)
    ax.scatter([x], [y], [z], color="crimson", s=40)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(
        title or f"Qubit {qubit} Bloch vector: ({x:.2f}, {y:.2f}, {z:.2f})"
    )
    ax.set_axis_off()
    return ax


def plot_amplitudes(
    state: np.ndarray,
    title: str = "State amplitudes",
    ax=None,
    kind: str = "probability",
):
    """Bar chart of state amplitudes.

    kind : 'probability' plots |amp|^2; 'real' plots Re(amp); 'complex' plots
    two bars per basis state (real + imag).
    """
    n = int(np.log2(len(state)))
    labels = [format(i, f"0{n}b") for i in range(len(state))]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(state)), 4))

    if kind == "probability":
        ax.bar(labels, np.abs(state) ** 2, color="#4c72b0", edgecolor="black")
        ax.set_ylabel("|amplitude|^2")
    elif kind == "real":
        ax.bar(labels, np.real(state), color="#4c72b0", edgecolor="black")
        ax.set_ylabel("Re(amplitude)")
    elif kind == "complex":
        width = 0.4
        xs = np.arange(len(state))
        ax.bar(xs - width / 2, np.real(state), width, label="Re", color="#4c72b0")
        ax.bar(xs + width / 2, np.imag(state), width, label="Im", color="#dd8452")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylabel("amplitude")
    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    ax.set_xlabel("Basis state")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return ax