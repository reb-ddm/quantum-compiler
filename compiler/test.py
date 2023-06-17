"""
Contains a few methods which are useful for testing.

It contains methods that transform all gates to matrices 
that can be used to check for equivalence between two circuits.
"""
import math
from .circuit import Circuit
from .optimization import EPS
import numpy as np

S = np.array([[1, 0], [0, 1j]])
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
SX = np.array([[(1+1j)/2, (1-1j)/2], [(1-1j)/2, (1+1j)/2]])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
ID = np.identity(2)

SDG = np.array([[1, 0], [0, -1j]])
SXDG = np.array([[(1-1j)/2, (1+1j)/2], [(1+1j)/2, (1-1j)/2]])
TDG = np.array([[1, 0], [0, np.exp(-1j*np.pi/4)]])


def zero_list_with_a_one(n, i):
    output = np.zeros(n)
    output[i] = 1
    return output


def print_matrix(U):
    """
    Print each entry of the matrix rounded to two decimal points.
    """
    string_matrix = ""
    for row in U:
        for element in row:
            string_matrix += str(np.round(element, 2))
            string_matrix += " "
        string_matrix += "\n"
    return string_matrix


def controlled_gate(U):
    n = len(U)
    controlled_gate = []
    for j in range(n):
        controlled_gate.append(np.array(zero_list_with_a_one(2 * n, j)))
    for j in range(n):
        controlled_gate.append(np.concatenate((np.zeros(n), U[j])))
    return controlled_gate


def n_controlled_gate(U, n):
    result_gate = U
    for _ in range(n):
        result_gate = controlled_gate(result_gate)
    return result_gate


CX = controlled_gate(X)

def gate_to_matrix(gate):
    # single qubit gate
    if gate.name == 'u':
        return U(gate.params[0], gate.params[1], gate.params[2], gate.params[3])
    elif gate.name == 'x':
        return X
    elif gate.name == 'y':
        return Y
    elif gate.name == 'z':
        return Z
    elif gate.name == 'h':
        return H
    elif gate.name == 'rz':
        return rz(gate.params[0])
    elif gate.name == 'ry':
        return ry(gate.params[0])
    elif gate.name == 'rx':
        return rx(gate.params[0])
    elif gate.name == 's':
        return S
    elif gate.name == 'sdg':
        return SDG
    elif gate.name == 't':
        return T
    elif gate.name == 'tdg':
        return TDG
    elif gate.name == 'sx':
        return SX
    elif gate.name == 'sxdg':
        return SXDG
    elif gate.name == 'id':
        return ID
    elif gate.name == 'ssx' and gate.nr_qubits() == 1:
        return ssx(gate.params[0])
    # two qubit gates
    elif gate.name == 'cy':
        return controlled_gate(Y)
    elif gate.name == 'cz':
        return controlled_gate(Z)
    elif gate.name == 'cs':
        return controlled_gate(S)
    elif gate.name == 'ct':
        return controlled_gate(T)
    elif gate.name == 'swap':
        return SWAP
    elif gate.name == 'crx':
        return controlled_gate(rx(gate.params[0]))
    elif gate.name == 'cu':
        return controlled_gate(U(gate.params[0], gate.params[1], gate.params[2], gate.params[3]))
    # arbitrary qubit gates
    elif gate.name == 'cx':
        return n_controlled_gate(X, gate.nr_qubits()-1)
    elif gate.name == 'ssx':
        return n_controlled_gate(ssx(gate.params[0]), gate.nr_qubits()-1)


def swap(permutation, i, j):
    temp = permutation[i]
    # find where j was
    for k in range(len(permutation)):
        if permutation[k] == j:
            permutation[k] = temp
    permutation[i] = j


def gate_to_matrix_n_qubits(gate, n):
    if gate.nr_qubits() == 1:
        # kronecker product with identity matrix for all qubits that are not involved in the gate
        result = np.array([1])
        for j in range(n):
            if j == gate.target():
                result = np.kron(result, gate_to_matrix(gate))
            else:
                result = np.kron(result, ID)
        return result
    if gate.nr_qubits() > 1:
        result = np.array([1])
        for j in range(n - gate.nr_qubits()):
            result = np.kron(result, ID)
        matrix = gate_to_matrix(gate)
        result = np.kron(result, matrix)
        permutation = [i for i in range(n)]
        swap(permutation, gate.target(), n-1)
        for (i, q) in enumerate(gate.control()):
            swap(permutation, q, n-i-2)
        result = reorder_gate(result, permutation)
        return result


def circuit_to_matrix(circuit):
    n = circuit.nr_qubits
    result = np.identity(2**n)
    for gate in circuit.gates:
        result = gate_to_matrix_n_qubits(gate, n) @ result
    return result


def reorder_gate(G, perm):
    """
    Credits: 
    This function (and the next three rotation matrix functions)
    are taken from a homework exercise template of the lecture
    "Advanced Concepts of Quantum Computing" at TUM in SS23.

    Adapt gate ’G’ to an ordering of the qubits as specified in ’perm’.
    Example, given G = np.kron(np.kron(A, B), C):
    reorder_gate(G, [1, 2, 0]) == np.kron(np.kron(B, C), A)
    """
    perm = list(perm)
    # number of qubits
    n = len(perm)
    # reorder both input and output dimensions
    perm2 = perm + [n + i for i in perm]
    return np.reshape(np.transpose(np.reshape(G, 2*n*[2]), perm2), (2**n, 2**n))


def rx(phi):
    """Returns a rotation matrix Rx(phi)"""
    return np.array([[np.cos(phi/2), -1j*np.sin(phi/2)],
                     [-1j*np.sin(phi/2), np.cos(phi/2)]])


def ry(phi):
    """Returns a rotation matrix Ry(phi)"""
    return np.array([[np.cos(phi/2), -np.sin(phi/2)],
                     [np.sin(phi/2), np.cos(phi/2)]])


def rz(phi):
    """Returns a rotation matrix Rz(phi)"""
    return np.array([[np.exp(-1j*phi/2), 0],
                     [0, np.exp(1j*phi/2)]])


def U(theta, phi, lam, rho):
    return np.exp(1j * rho) * np.array([
        [np.cos(theta/2), -np.exp(lam*1j) * np.sin(theta/2)],
        [np.exp(phi*1j) * np.sin(theta/2),
         np.exp((phi + lam)*1j) * np.cos(theta/2)]
    ])


def ssx(k):
    if k >= 0:
        return U(np.pi/2**k, -np.pi/2, np.pi/2, np.pi/2**(k+1))
    else:
        k = np.abs(k)
        return U(-np.pi/2**(k), -np.pi/2, np.pi/2, -np.pi/2**(k+1))


def snd(U):
    return np.kron(id, U)


def fst(U):
    return np.kron(U, id)


def areEqual(matrix1, matrix2):
    """
    Returns true if the distance between the two matrices is less than EPS.
    """
    return np.linalg.norm(matrix1 - matrix2) < EPS


def find_phase(matrix1, matrix2):
    """
    If the two matrices are equal up to a phase shift, then `matrix1=phase*matrix2` and this 
    function outputs the corresponding phase by finding the first non-zero entry and then dividing the
    entries of the two matrices.
    """
    i = 0
    j = 0
    n = len(matrix1)
    while i < n and matrix2[i][0] == 0:
        j = 0
        while j < n and matrix2[i][j] == 0:
            if math.isnan(matrix2[i][j].real):
                print("matrix 2" + str(i) + " " + str(j))
            if math.isnan(matrix1[i][j].real):
                print("matrix 1" + str(i) + " " + str(j))
            j += 1
        if matrix2[i][j] == 0:
            i += 1
        else:
            break
    if j >= n or i >= n:
        return 1
    return matrix1[i][j]/matrix2[i][j]


def are_equal_ignore_phase(matrix1, matrix2):
    phase = find_phase(matrix1, matrix2)
    return areEqual(matrix1, phase*matrix2)


