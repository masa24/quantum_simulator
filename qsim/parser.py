"""Parser for a subset of OpenQASM 2.0.

Handles the gate set IBM Quantum emits: qreg/creg declarations,
standard gates from qelib1.inc, and terminal measurements.
Does NOT handle: custom gate definitions, classical conditionals,
mid-circuit measurements with feedback, or OpenQASM 3 syntax.
"""
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Instruction:
    """A single circuit instruction."""
    name: str                          # gate name, e.g. "h", "cx", "measure"
    qubits: list[int] = field(default_factory=list)
    clbits: list[int] = field(default_factory=list)
    params: list[float] = field(default_factory=list)


@dataclass
class Circuit:
    """A parsed quantum circuit."""
    num_qubits: int = 0
    num_clbits: int = 0
    instructions: list[Instruction] = field(default_factory=list)


def _eval_param(expr: str) -> float:
    """Evaluate a parameter expression like 'pi/2' or '3*pi/4'.

    Only allows pi, numbers, and + - * / ( ). No arbitrary code execution.
    """
    expr = expr.strip()
    if not re.fullmatch(r"[\d\.piePI\s\+\-\*/\(\)]+", expr):
        raise ValueError(f"Unsafe parameter expression: {expr!r}")
    # Allow only 'pi' as a name.
    return float(eval(expr, {"__builtins__": {}}, {"pi": 3.141592653589793}))


def _parse_qubit_ref(ref: str) -> int:
    """Extract the index from a reference like 'q[3]'."""
    m = re.fullmatch(r"\s*\w+\s*\[\s*(\d+)\s*\]\s*", ref)
    if not m:
        raise ValueError(f"Cannot parse qubit reference: {ref!r}")
    return int(m.group(1))


def parse_qasm(source: str) -> Circuit:
    """Parse OpenQASM 2.0 source code into a Circuit."""
    circuit = Circuit()

    # Strip comments and split on semicolons.
    source = re.sub(r"//.*", "", source)
    statements = [s.strip() for s in source.split(";") if s.strip()]

    for stmt in statements:
        # Skip header / include lines.
        if stmt.startswith("OPENQASM") or stmt.startswith("include"):
            continue

        # qreg q[n]
        m = re.fullmatch(r"qreg\s+\w+\s*\[\s*(\d+)\s*\]", stmt)
        if m:
            circuit.num_qubits = int(m.group(1))
            continue

        # creg c[n]
        m = re.fullmatch(r"creg\s+\w+\s*\[\s*(\d+)\s*\]", stmt)
        if m:
            circuit.num_clbits = int(m.group(1))
            continue

        # measure q[i] -> c[j]
        m = re.fullmatch(
            r"measure\s+(\w+\s*\[\s*\d+\s*\])\s*->\s*(\w+\s*\[\s*\d+\s*\])", stmt
        )
        if m:
            q_idx = _parse_qubit_ref(m.group(1))
            c_idx = _parse_qubit_ref(m.group(2))
            circuit.instructions.append(
                Instruction(name="measure", qubits=[q_idx], clbits=[c_idx])
            )
            continue

        # barrier (ignore)
        if stmt.startswith("barrier"):
            continue

        # Gate: optional params in parens, then comma-separated qubit refs.
        # Examples: "h q[0]", "cx q[0], q[1]", "rx(pi/2) q[0]", "u3(pi/2,0,pi) q[1]"
        m = re.fullmatch(
            r"(\w+)\s*(?:\(([^)]*)\))?\s+(.+)",
            stmt,
        )
        if not m:
            raise ValueError(f"Cannot parse statement: {stmt!r}")

        name = m.group(1).lower()
        param_str = m.group(2)
        qubit_list_str = m.group(3)

        params = []
        if param_str is not None and param_str.strip():
            params = [_eval_param(p) for p in param_str.split(",")]

        qubits = [_parse_qubit_ref(q) for q in qubit_list_str.split(",")]

        circuit.instructions.append(
            Instruction(name=name, qubits=qubits, params=params)
        )

    return circuit


def parse_qasm_file(path: str) -> Circuit:
    """Parse a QASM file from disk."""
    with open(path, "r") as f:
        return parse_qasm(f.read())