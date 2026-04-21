// Bell state: (|00> + |11>) / sqrt(2)
// Expected: 50/50 between '00' and '11', never '01' or '10'.
// The simplest demonstration of entanglement.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

h q[0];
cx q[0], q[1];

measure q[0] -> c[0];
measure q[1] -> c[1];
