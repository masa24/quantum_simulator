// Deutsch-Jozsa algorithm with a balanced function f: {0,1}^3 -> {0,1}.
// For this circuit, f(x) = x[0] XOR x[1] XOR x[2] (parity - balanced).
// Qubits 0,1,2 are the input register, qubit 3 is the output register.
//
// Expected: measuring the input register NEVER returns '000'.
// This one-shot answer distinguishes balanced from constant functions.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[3];

// Prepare output register in |-> = (|0> - |1>)/sqrt(2)
x q[3];
h q[3];

// Superposition over input register.
h q[0];
h q[1];
h q[2];

// Oracle for f(x) = x0 XOR x1 XOR x2: just CNOT each input into output.
cx q[0], q[3];
cx q[1], q[3];
cx q[2], q[3];

// Final Hadamards on input register.
h q[0];
h q[1];
h q[2];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
