// W state: (|100> + |010> + |001>) / sqrt(3)
// Expected: roughly 1/3 on each of '100', '010', '001'; never '000'.
// Interesting contrast to GHZ - entangled in a completely different way.
// The W state's entanglement survives losing one qubit; GHZ's does not.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Rotation angles chosen so the amplitudes split evenly into thirds.
ry(1.9106332362490184) q[0];
ch q[0], q[1];
ccx q[0], q[1], q[2];
x q[0];
cx q[0], q[1];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
