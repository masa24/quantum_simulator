// QFT followed by inverse QFT = identity.
// Apply QFT to |011>, then inverse QFT, measure — should always give '011'.
// Great test that your simulator correctly handles parameterized two-qubit
// phase gates and that gate sequences compose correctly.
//
// Expected: every shot measures '011'.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Prepare |011>: flip qubits 0 and 1.
x q[0];
x q[1];

// --- 3-qubit QFT ---
h q[2];
cu1(1.5707963267948966) q[1], q[2];
cu1(0.7853981633974483) q[0], q[2];

h q[1];
cu1(1.5707963267948966) q[0], q[1];

h q[0];

swap q[0], q[2];

// --- 3-qubit inverse QFT (reverse order, negate angles) ---
swap q[0], q[2];

h q[0];

cu1(-1.5707963267948966) q[0], q[1];
h q[1];

cu1(-0.7853981633974483) q[0], q[2];
cu1(-1.5707963267948966) q[1], q[2];
h q[2];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
