from compiler import Circuit, circuit_to_matrix, are_equal_ignore_phase, Gate, reorder_gate, ID

import unittest
import numpy as np

# a few concrete qubit architectures for the tests
small_3_qubits = [[0, 2], [2, 0], [1, 2], [2, 1]]
ibmq_manila_5_qubits = [[0, 1], [1, 0], [1, 2],
                        [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]]
ibmq_lima_5_qubits = [[0, 1], [1, 0], [1, 2],
                      [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
ibm_perth_7_qubits = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [
    3, 1], [3, 5], [4, 5], [5, 3], [5, 4], [5, 6], [6, 5]]
large_9_qubits = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3], [
    4, 5], [5, 4], [6, 0], [0, 6], [6, 5], [5, 6], [2, 7], [7, 2], [2, 8], [8, 2]]


def inverse_mapping(mapping):
    """
    A mapping is basically a permutation. This function finds the inverse permutation by inverting all cycles of the permutation.
    """
    n = len(mapping)
    already_mapped = [False for _ in range(n)]
    final_mapping = [0 for _ in range(n)]
    for i in range(n):
        if already_mapped[i] == False:
            next = mapping[i]
            # find next cycle
            cycle = [next]
            while (next != i):
                next = mapping[next]
                cycle.append(next)
            # add opposite cycle to final mapping
            j = len(cycle) - 1
            current_element = cycle[j]
            done = False
            while not done:
                j -= 1
                if j == -1:
                    j = len(cycle) - 1
                    done = True
                final_mapping[current_element] = cycle[j]
                already_mapped[current_element] = True
                current_element = cycle[j]
    return final_mapping


def concat_mapping(mapping1, mapping2):
    """
    Returns a permutation which is the composition of the two permutations.
    """
    n = len(mapping1)
    final_mapping = [0 for _ in range(n)]
    for i in range(n):
        final_mapping[i] = mapping2[mapping1[i]]
    return final_mapping


def swaps_for_mapping(mapping):
    """
    Computes which swaps need to be performed to arrive to this mapping.
    """
    n = len(mapping)
    already_mapped = [False for _ in range(n)]
    final_swaps = []
    for i in range(n):
        if already_mapped[i] == False:
            next = mapping[i]
            # find next cycle
            cycle = [[i, next]]
            while (next != i):
                next_next = mapping[next]
                cycle.append([next, next_next])
                next = next_next
            # compute swaps for each cycle separately
            j = len(cycle) - 1
            current_element = cycle[j]
            while j > 0:
                j -= 1
                final_swaps.append(Gate('swap', current_element))
                already_mapped[current_element[0]] = True
                already_mapped[current_element[1]] = True
                current_element = cycle[j]
    return final_swaps


def add_swaps_to_get_to_mapping(initial_mapping, final_mapping):
    """
    Computes which swaps need to be performed to arrive to the `final_mapping`
    when we are currently in the `initial_mapping`.
    """
    inverse_final = inverse_mapping(final_mapping)
    total_mappping = concat_mapping(inverse_final, initial_mapping)
    return swaps_for_mapping(total_mappping)


def route_and_compare(circuit, coupling_map, ph_qubits, virt_qubits):
    matrix1 = circuit_to_matrix(circuit)

    initial_mapping, final_mapping = circuit.route(coupling_map, ph_qubits)

    # extend mapping to make it a permutation
    not_used_ph_qubits_final = [q for q in range(
        ph_qubits) if q not in final_mapping]
    not_used_ph_qubits_initial = [q for q in range(
        ph_qubits) if q not in initial_mapping]

    initial_mapping, final_mapping = initial_mapping + \
        not_used_ph_qubits_initial, final_mapping + not_used_ph_qubits_final
    # we add swap operations at the end in order to go back to the initial mapping at the end
    # so that the mapping is the same in the beginning and in the end
    circuit.gates += add_swaps_to_get_to_mapping(
        initial_mapping, final_mapping)

    matrix2 = circuit_to_matrix(circuit)

    # route() changes the number and order of qubits, so we need to reshape and reshuffle
    for _ in range(virt_qubits, ph_qubits):
        matrix1 = np.kron(matrix1, ID)

    matrix2 = reorder_gate(matrix2, initial_mapping)

    return are_equal_ignore_phase(matrix1, matrix2)


def all_cnot_valid(circuit, coupling_map):
    for gate in circuit.gates:
        if gate.name == 'cx' and gate.qubits not in coupling_map:
            return False
    return True


class TestRoutingEquivalence(unittest.TestCase):
    def test_small_circuit_small_architecture(self):
        circuit = Circuit(3)
        circuit.cnot(1, 2)
        circuit.cnot(1, 0)
        circuit.cnot(0, 2)
        self.assertTrue(route_and_compare(circuit, small_3_qubits, 3, 3))

    def test_small_circuit(self):
        circuit = Circuit(3)
        circuit.cnot(1, 2)
        circuit.cnot(1, 0)
        circuit.cnot(0, 2)
        self.assertTrue(route_and_compare(circuit, ibmq_manila_5_qubits, 5, 3))

    def test_very_small_circuit(self):
        circuit = Circuit(2)
        circuit.cnot(1, 0)
        circuit.cnot(0, 1)
        self.assertTrue(route_and_compare(circuit, small_3_qubits, 3, 2))

    def test_medium_circuit(self):
        circuit = Circuit(4)
        circuit.cnot(1, 2)
        circuit.cnot(1, 0)
        circuit.cnot(0, 2)
        circuit.cnot(0, 3)
        circuit.cnot(1, 2)
        circuit.cnot(1, 3)
        self.assertTrue(route_and_compare(circuit, ibmq_manila_5_qubits, 5, 4))
        self.assertTrue(route_and_compare(circuit, ibmq_lima_5_qubits, 5, 5))

    def test_big_circuit(self):
        circuit = Circuit(7)
        circuit.cnot(1, 2)
        circuit.cnot(1, 0)
        circuit.cnot(0, 2)
        circuit.cnot(0, 3)
        circuit.cnot(1, 2)
        circuit.cnot(1, 3)
        circuit.cnot(1, 6)
        circuit.cnot(4, 5)
        circuit.cnot(6, 5)
        circuit.cnot(4, 5)
        circuit.cnot(3, 5)
        circuit.cnot(0, 5)
        self.assertTrue(route_and_compare(circuit, large_9_qubits, 9, 7))


class TestRoutingValid(unittest.TestCase):
    def test_small_circuit(self):
        circuit = Circuit(3)
        circuit.cnot(1, 2)
        circuit.cnot(1, 0)
        circuit.cnot(0, 2)
        circuit.route(ibmq_manila_5_qubits, 5)
        self.assertTrue(all_cnot_valid(circuit, ibmq_manila_5_qubits))

    def test_big_circuit(self):
        circuit = Circuit(6)
        circuit.cnot(1, 2)
        circuit.s(2)
        circuit.cnot(1, 4)
        circuit.cnot(0, 2)
        circuit.x(2)
        circuit.cnot(5, 2)
        circuit.cnot(1, 0)
        circuit.cnot(0, 3)
        circuit.u(2, 1, 1, 1, 1)
        circuit.cnot(2, 5)
        circuit.cnot(1, 3)
        circuit.cnot(0, 4)
        circuit.route(ibm_perth_7_qubits, 7)
        self.assertTrue(all_cnot_valid(circuit, ibm_perth_7_qubits))

    def test_big_circuit2(self):
        circuit = Circuit(6)
        circuit.cnot(1, 2)
        circuit.s(2)
        circuit.cnot(1, 4)
        circuit.cnot(0, 2)
        circuit.x(2)
        circuit.cnot(5, 2)
        circuit.cnot(1, 0)
        circuit.cnot(0, 3)
        circuit.u(2, 1, 1, 1, 1)
        circuit.cnot(2, 5)
        circuit.cnot(4, 3)
        circuit.cnot(3, 5)
        circuit.cnot(1, 3)
        circuit.cnot(1, 4)
        circuit.cnot(4, 3)
        circuit.cnot(3, 5)
        circuit.cnot(1, 3)
        circuit.cnot(1, 4)
        circuit.cnot(4, 3)
        circuit.cnot(3, 5)
        circuit.cnot(1, 3)
        circuit.cnot(1, 4)
        circuit.route(large_9_qubits, 9)
        self.assertTrue(all_cnot_valid(circuit, large_9_qubits))
