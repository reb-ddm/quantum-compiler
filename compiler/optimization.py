"""
Functions for simple optimizations of quantum circuits.
"""
import numpy as np
from .gate import Gate, EPS


def optimize(gates, nr_qubits):
    gates = _remove_adjacent_inverse_gates(gates, nr_qubits)
    gates = _remove_null_rotations(gates)
    gates = _merge_rotations(gates, nr_qubits)
    gates = _remove_null_rotations(gates)
    return gates


def _remove_null_rotations(gates):
    """
    Removes all RZ, RY and RX rotations of angle `2*n*π` and ID operations.
    """
    optimized_gates = []
    for gate in gates:
        if (gate.name == 'rz' or gate.name == 'ry' or gate.name == 'rx') and abs((gate.params[0] % (2*np.pi)) < EPS or abs(gate.params[0] % (2*np.pi) - (2*np.pi)) < EPS):
            pass
        elif gate.name == 'id':
            pass
        else:
            optimized_gates += [gate]
    return optimized_gates


def _merge_rotations(gates, nr_qubits):
    """
    Merges all consecutive RZ rotations by changing them to ID gates, which can be removed by applying the `_remove_null_rotations()` function.
    """
    last_operation = [-1 for _ in range(
        nr_qubits)]  # records for each qubit at which index we saw the last rotation on this qubit
    optimized_gates = gates.copy()
    for i in range(len(gates)):
        # check for each gate if it is an 'rz' gate
        gate = gates[i]
        target = gate.target()
        index_last_operation = last_operation[target]
        if gate.name == 'rz':
            if index_last_operation == -1:
                last_operation[target] = i
            else:
                # the last operation on this qubit was a rotation rz, therefore we add the current rotation angle to the last rotation
                # and remove the current rotation
                # in order to not change the index of the previous gates, instead of removing the gate, we transform it to an 'id' gate,
                # which can be removed later by the function _remove_null_rotations
                new_rotation_angle = (
                    optimized_gates[index_last_operation].params[0] + gate.params[0]) % (2 * np.pi)
                optimized_gates[index_last_operation] = Gate(
                    'rz', target, params=[new_rotation_angle])
                optimized_gates[i] = Gate('id', target)
        else:
            last_operation[target] = -1
            for c in gate.control():
                last_operation[c] = -1
    return optimized_gates


def _remove_adjacent_inverse_gates(gates, nr_qubits):
    """
    Removes adjacent inverse gates by changing them to ID gates, 
    which can be removed by applying the `_remove_null_rotations()` function.
    """
    # records for each qubit at which index we saw the last operation on this qubit
    last_operation = [-1 for _ in range(nr_qubits)]
    optimized_gates = gates.copy()
    for i in range(len(gates)):
        # check for each gate if it is inverse to the last operation on this qubit
        gate = gates[i]
        target = gate.target()
        index_last_operation = last_operation[target]
        if index_last_operation == -1:
            last_operation[target] = i
        else:
            if gate.is_inverse(gates[index_last_operation]) and (gate.nr_qubits() == 1 or last_operation[gate.control()[0]] < index_last_operation) and gate.nr_qubits() < 3:
                # the last operation on this qubit was an inverse operation to the current operation, therefore we remove the two operations from the output
                # in order to not change the index of the previous gates, instead of removing the gates, we transform it to an 'id' gate,
                # which can be removed later by the function _remove_null_rotations
                optimized_gates[index_last_operation] = Gate('id', target)
                optimized_gates[i] = Gate('id', target)
                last_operation[target] = -1
                # we don't remember which one the was the last operation before the one we deleted.
                # therefore it could be useful to run this optimization several times
            else:
                last_operation[target] = i
        for c in gate.control():
            last_operation[c] = -1
    return optimized_gates
