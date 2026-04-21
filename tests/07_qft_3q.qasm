// 3-qubit Quantum Fourier Transform applied to |001>.
// The QFT of a computational basis state |x> gives a uniform-magnitude
// superposition with phases encoding x.
//
// Expected: all 8 basis states measured with roughly equal probability
// (~12.5% each) — any input to QFT produces a uniform magnitude distribution.
// The information is in the *phases*, which measurement destroys.
// To verify the phase structure, you'd need to run QFT then inverse-QFT
// and measure — see circuit 08.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Prepare input state |001>: flip qubit 0.
x q[0];

// --- 3-qubit QFT ---
h q[2];
cu1(1.5707963267948966) q[1], q[2];   // controlled-R_2 (pi/2)
cu1(0.7853981633974483) q[0], q[2];   // controlled-R_3 (pi/4)

h q[1];
cu1(1.5707963267948966) q[0], q[1];   // controlled-R_2 (pi/2)

h q[0];

// Swap qubits 0 and 2 to match standard QFT output convention.
swap q[0], q[2];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
