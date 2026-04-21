// Grover's algorithm for 2 qubits, marked state |11>.
// With a single Grover iteration on 2 qubits, the marked state is found
// with probability 1 (the algorithm is actually exact for n=2).
//
// Expected: all 1024 shots should measure '11'.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

// Initialize uniform superposition.
h q[0];
h q[1];

// Oracle marking |11>: apply CZ (equivalent to phase flip on |11>).
cz q[0], q[1];

// Diffusion operator: H . (2|0><0| - I) . H
h q[0];
h q[1];
x q[0];
x q[1];
cz q[0], q[1];
x q[0];
x q[1];
h q[0];
h q[1];

measure q[0] -> c[0];
measure q[1] -> c[1];
