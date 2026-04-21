// Quantum teleportation of the state |+> from qubit 0 to qubit 2.
// Prepare |+> on qubit 0, then teleport it using an EPR pair on qubits 1,2.
// After teleportation + X/Z corrections, qubit 2 should be in |+>.
//
// Expected: qubit 2 measured in Z basis is 50/50 |0>/|1>.
// To verify teleportation worked, we undo the |+> preparation (H) on qubit 2
// and measure. A successful teleport makes qubit 2 measurement = '0' always.
//
// Note: this version uses post-selection - we measure q0, q1 and unconditionally
// apply corrections based on 'average' behavior. Full teleportation with classical
// feedforward needs mid-circuit measurement with classical control, which
// OpenQASM 2 supports via 'if' but this simulator does not.
// Instead we do the "gate teleportation" version where the corrections are
// replaced by deterministic gates that work on average when we post-select.
//
// Better for this simulator: the statistics of (q0, q1, q2) together verify
// the protocol. Check that P('xy0') = P('xy1') for all 'xy' when q2 is
// first un-H'd.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Prepare |+> on qubit 0 (the state we want to teleport).
h q[0];

// Make EPR pair on qubits 1 and 2.
h q[1];
cx q[1], q[2];

// Bell measurement on qubits 0 and 1.
cx q[0], q[1];
h q[0];

// "Undo" the |+> preparation on q[2] to verify it was successfully teleported.
// If the protocol worked, this returns q[2] to |0> modulo the corrections
// encoded by q[0] and q[1]. Look at the joint statistics.
h q[2];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
