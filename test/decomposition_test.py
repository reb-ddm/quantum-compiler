import unittest
import numpy as np

from compiler import Circuit, circuit_to_matrix, are_equal_ignore_phase, Gate, native_gates


def decompose_and_compare(circuit):
    matrix1 = circuit_to_matrix(circuit)
    circuit.decompose()
    matrix2 = circuit_to_matrix(circuit)
    return are_equal_ignore_phase(matrix1, matrix2)


def partially_decompose_and_compare(circuit, degree_of_decomposition):
    matrix1 = circuit_to_matrix(circuit)
    gates = circuit.gates
    if degree_of_decomposition > 0:
        # decompose to 2-qubit gates
        decomposed_gates = []
        for gate in gates:
            decomposed_gates += gate._decompose_multiple_qubit_gates()
        gates = decomposed_gates
        if degree_of_decomposition > 1:
            # decompose to CNOT and U
            decomposed_gates = []
            for gate in gates:
                decomposed_gates += gate._decompose_to_cx_rz_u()
            gates = decomposed_gates
            if degree_of_decomposition > 2:
                # decompose U to ['cx', 'id', 'rz', 'x', 'sx']
                decomposed_gates = []
                for gate in gates:
                    decomposed_gates += gate._decompose_u_gate()
                gates = decomposed_gates
    circuit.gates = gates
    matrix2 = circuit_to_matrix(circuit)
    return are_equal_ignore_phase(matrix1, matrix2)


def only_contains_gates_from_set(gates, gate_set):
    for gate in gates:
        if gate.name not in gate_set:
            return False
        if gate.nr_qubits() > 2:
            return False
    return True


def large_circuit():
    """circuit with 5 qubits containing at least one instance of each supported gate"""
    circuit = Circuit(5)
    circuit.u(1, np.pi, 4, 2.45, 0)
    circuit.x(0)
    circuit.y(0)
    circuit.z(1)
    circuit.h(3)
    circuit.rz(2, np.pi/5)
    circuit.ry(1, np.pi/7)
    circuit.rx(0, 2)
    circuit.s(4)
    circuit.sdg(3)
    circuit.t(1)
    circuit.tdg(2)
    circuit.sx(1)
    circuit.sxdg(0)
    circuit.id(3)
    circuit.cx(0, 1)
    circuit.cy(0, 1)
    circuit.cz(0, 1)
    circuit.cs(2, 1)
    circuit.ct(0, 3)
    circuit.cu(3, 4, np.pi, 4, 2.45, 0)
    circuit.crx(3, 4, np.pi)
    circuit.swap(2, 4)
    circuit.ccx(1, 2, 3)
    circuit.ssx(3, [0, 1, 4], 3)
    circuit.ssx(2, [0, 3], -3)
    return circuit


class TestDecompositionEquivalence(unittest.TestCase):
    """Test if the decomposed circuit is equivalent to the original circuit
    by converting the circuit to its matrix representation."""

    def test_decompose(self):
        """full decomposition of some example circuits"""
        ry = Circuit(1)
        ry.ry(0, 1)
        self.assertTrue(decompose_and_compare(ry))
        ssx = Circuit(3)
        ssx.gates = [Gate('ssx', 2, [-1])]
        self.assertTrue(decompose_and_compare(ssx))
        toffoli = Circuit(3)
        toffoli.ccx(0, 1, 2)
        self.assertTrue(decompose_and_compare(toffoli))
        u = Circuit(3)
        u.u(0, 1, 2, 3, 4)
        u.ry(1, np.pi/5)
        u.u(1, np.pi, np.pi/2, np.pi/2, np.pi)
        self.assertTrue(decompose_and_compare(u))
        cu = Circuit(4)
        cu.cu(0, 1, 1, 2, 3, 4)
        cu.cu(2, 3, np.pi, np.pi/2, np.pi/2, np.pi)
        self.assertTrue(decompose_and_compare(cu))

    def test_decompose_large_circuit(self):
        """decomposition of circuit containing all the supported gates"""
        self.assertTrue(decompose_and_compare(large_circuit()))

    def test_decompose_multiple_qubit_gates(self):
        """decomposition of multiple qubit gates (> 2 qubits) to 2-qubit gates"""
        toffoli = Circuit(3)
        toffoli.ccx(0, 1, 2)
        self.assertTrue(partially_decompose_and_compare(toffoli, 1))
        toffoli3 = Circuit(5)
        toffoli3.gates = [Gate('cx', [4, 0, 1, 2, 3])]
        self.assertTrue(partially_decompose_and_compare(toffoli3, 1))

    def test_decompose_to_cx_rz_u(self):
        """decomposition of gates to CNOT and single qubit gates"""
        ssx = Circuit(3)
        ssx.gates = [Gate('ssx', 2, [1])]
        self.assertTrue(partially_decompose_and_compare(ssx, 2))
        cssx = Circuit(3)
        cssx.gates = [Gate('ssx', [2, 0], [-1])]
        self.assertTrue(partially_decompose_and_compare(cssx, 2))

    def test_decompose_u_gate(self):
        """decomposition of U gate to the native gates ['rz', 'x', 'sx']"""
        u = Circuit(3)
        u.u(2, 1, 2, 3, 4)
        self.assertTrue(partially_decompose_and_compare(u, 3))


class TestDecompositionValid(unittest.TestCase):
    """Test if the decomposed circuit is really decomposed, i.e. it contains only contains
    the following gates: ['cx', 'id', 'rz', 'x', 'sx']."""

    def test_decomposed_native_gates(self):
        circuit = large_circuit().decompose()
        self.assertTrue(only_contains_gates_from_set(
            circuit.gates, native_gates))


if __name__ == '__main__':
    unittest.main()
