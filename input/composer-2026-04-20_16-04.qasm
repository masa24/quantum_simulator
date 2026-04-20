OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];
h q[0];
tdg q[1];
s q[2];
rz(pi/2) q[1];
