"""
Representation of a quantum circuit.
"""
import numpy as np
import qiskit

from .gate import Gate, NAME_LENGTH, native_gates, pad_to_length
from .routing import route
from .optimization import optimize


def to_adjacency_list(coupling_map, nr_qubits):
    """
    Converts coupling map to adjacency list.
    A coupling map is a list of pairs of connected qubits.
    An adjacency list contains at index i a list of qubits with which the qubit i is connected.
    """ 
    adjacent_qubits = [[] for _ in range(nr_qubits)]

    for pair in coupling_map:
        adjacent_qubits[pair[0]].append(pair[1])
    return adjacent_qubits


class Circuit:

    def __init__(self, nr_qubits):
        self.nr_qubits = nr_qubits
        self.gates = []
        self.decomposed = True
        self.nr_measurements = 0

    # compilation functions

    def compile(self, coupling_map, nr_qubits):
        self.optimize()
        self.decompose()
        self.optimize()
        self.route(coupling_map, nr_qubits)
        self.optimize()

    def decompose(self):
        """
        Decomposes the circuit to an equivalent circuit which contains only the gates ['cx', 'id', 'rz', 'x', 'sx'].
        """  
        decomposed_gates = []
        for gate in self.gates:
            decomposed_gates += gate.decompose()
        self.gates = decomposed_gates
        self.decomposed = True
        return self

    def route(self, coupling_map, nr_qubits):  
        """
        Maps the gates of this circuit to a physical architecture described by the `coupling_map`.
        The number of qubits in this circuit is changed to `nr_qubits`.
        This function reorders the qubits and adds swap gates such that each CNOT operation is performed 
        between adjacent qubits. 

        Parameters:
            The `coupling_map` describes which qubits are adjacent. It's a list of pairs, 
            where each pair represents a connection between two qubits.

            `nr_qubits` defines how many qubits are available in the physical quantum computer.

        Returns:
            `initial_mapping`:  The mapping from the virtual qubits to the physical qubits they are represented by at the beginning of the routed circuit.
            `mapping[i]` contains the physical qubits to which the virtual qubit `i` is mapped at the end of the circuit.

            `mapping`: The mapping from the virtual qubits to the physical qubits they are represented by at the end of the routed circuit.
            
        """
        if self.nr_qubits > nr_qubits:
            raise Exception(
                f"There are only {nr_qubits} physical qubits available. The circuit needs at least {self.nr_qubits} qubits.")

        adjacent_qubits = to_adjacency_list(coupling_map, nr_qubits)
        initial_mapping, mapping, self.gates = route(
            nr_qubits, self.nr_qubits, self.gates, adjacent_qubits)
        self.nr_qubits = nr_qubits
        return initial_mapping, mapping

    def optimize(self):
        """
        Performs a few simple optimizations, if possible.
        """
        self.gates = optimize(self.gates, self.nr_qubits)
        return self
    
    # functions to add gates to the circuit

    # single qubit gates
    def __one_qubit_gate(self, qubit, name, params=[]):
        self.__out_of_range_check(qubit)
        self.gates.append(Gate(name, qubit, params=params))
        if name not in native_gates:
            self.decomposed = False

    def u(self, qubit, theta, phi, lam, rho):
        self.__one_qubit_gate(qubit, 'u', params=[theta % (
            2*np.pi), phi % (2*np.pi), lam % (2*np.pi), rho % (2*np.pi)])

    def x(self, qubit):
        self.__one_qubit_gate(qubit, 'x')

    def y(self, qubit):
        self.__one_qubit_gate(qubit, 'y')

    def z(self, qubit):
        self.__one_qubit_gate(qubit, 'z')

    def h(self, qubit):
        self.__one_qubit_gate(qubit, 'h')

    def rz(self, qubit, phi):
        self.__one_qubit_gate(qubit, 'rz', params=[phi % (2*np.pi)])

    def ry(self, qubit, phi):
        self.__one_qubit_gate(qubit, 'ry', params=[phi % (2*np.pi)])

    def rx(self, qubit, phi):
        self.__one_qubit_gate(qubit, 'rx', params=[phi % (2*np.pi)])

    def s(self, qubit):
        self.__one_qubit_gate(qubit, 's')

    def sdg(self, qubit):
        """Inverse of the S gate"""
        self.__one_qubit_gate(qubit, 'sdg')

    def t(self, qubit):
        self.__one_qubit_gate(qubit, 't')

    def tdg(self, qubit):
        """Inverse of the T gate"""
        self.__one_qubit_gate(qubit, 'tdg')

    def sx(self, qubit):
        self.__one_qubit_gate(qubit, 'sx')

    def v(self, qubit):
        self.sx(qubit)

    def sxdg(self, qubit):
        """Inverse of the SX gate"""
        self.__one_qubit_gate(qubit, 'sxdg')

    def vdg(self, qubit):
        """Inverse of the V (also called SX) gate"""
        self.sxdg(qubit)
    
    def id(self, qubit):
        self.__one_qubit_gate(qubit, 'id')

    # two qubit gates
    def __two_qubit_gate(self, control, target, name, params=[]):
        self.__out_of_range_check(control)
        self.__out_of_range_check(target)
        self.gates.append(Gate(name, [target, control], params=params))
        if name not in native_gates:
            self.decomposed = False

    def cx(self, control, target):
        self.__two_qubit_gate(control, target, 'cx')

    def cnot(self, control, target):
        self.cx(control, target)

    def cy(self, control, target):
        self.__two_qubit_gate(control, target, 'cy')

    def cz(self, control, target):
        self.__two_qubit_gate(control, target, 'cz')
    
    def cs(self, control, target):
        self.__two_qubit_gate(control, target, 'cs')
    
    def ct(self, control, target):
        self.__two_qubit_gate(control, target, 'ct')

    def crx(self, control, target, phi):
        """Controlled X rotation gate"""
        self.__two_qubit_gate(control, target, 'crx', params=[phi % (2*np.pi)])

    def cu(self, control, target, theta, phi, lam, rho):
        """Controlled U gate"""
        self.__two_qubit_gate(control, target, 'cu', params=[theta % (
            2*np.pi), phi % (2*np.pi), lam % (2*np.pi), rho % (2*np.pi)])

    def swap(self, control, target):
        self.__two_qubit_gate(control, target, 'swap')

    # three qubit gates
    def __three_qubit_gate(self, control1, control2, target, name):
        self.__out_of_range_check(control1)
        self.__out_of_range_check(control2)
        self.__out_of_range_check(target)
        self.gates.append(Gate(name, [target, control2, control1]))
        self.decomposed = False

    def ccx(self, control1, control2, target):
        self.__three_qubit_gate(control1, control2, target, 'cx')

    def toffoli(self, control1, control2, target):
        self.ccx(control1, control2, target)

    # multiple qubit gates
    def __multiple_qubit_gate(self, controls, target, name, params=[]):
        if len(controls) == 0 and name != 'ssx':
            raise Exception("There must be at least one control qubit.")
        for control in controls:
            self.__out_of_range_check(control)
        self.__out_of_range_check(target)
        self.gates.append(Gate(name, [target] + controls, params=params))
        if name != 'cx' or len(controls) > 1:
            self.decomposed = False

    def ssx(self, target, controls, k):
        self.__multiple_qubit_gate(controls, target, 'ssx', params=[k])

    def mcx(self, target, controls):
        """Multiple-controlled X gate"""
        self.__multiple_qubit_gate(controls, target, 'cx')

    # measurement and reset
    def measure(self, qubits):
        if isinstance(qubits, list):
            for qubit in qubits:
                self.__out_of_range_check(qubit)
                self.__one_qubit_gate(qubit, 'measure')
                self.nr_measurements += 1
        else:
            self.__out_of_range_check(qubits)
            self.__one_qubit_gate(qubits, 'measure')
            self.nr_measurements += 1

    def reset(self, qubits):
        if isinstance(qubits, list):
            for qubit in qubits:
                self.__out_of_range_check(qubit)
                self.__one_qubit_gate(qubit, 'reset')
        else:
            self.__out_of_range_check(qubits)
            self.__one_qubit_gate(qubits, 'reset')

    # function for drawing and simulating using qiskit

    def to_qiskit_circuit(self):
        circ = qiskit.QuantumCircuit(self.nr_qubits, self.nr_measurements)
        current_measurement_index = 0
        for gate in self.gates:
            qiskit_gate = gate.to_qiskit_gate()
            if gate.name == 'measure':
                circ.append(qiskit_gate, gate.qubits, [
                            current_measurement_index])
                current_measurement_index += 1
            else:
                circ.append(qiskit_gate, gate.control() + [gate.target()])
        return circ

    def qiskit_draw(self):
        return self.to_qiskit_circuit().draw()

    def qiskit_draw2(self):
        return self.to_qiskit_circuit().draw('mpl')

    def simulate(self):
        if self.nr_measurements == 0:
            raise Exception(
                "You need at least one measurement instruction in order to run the circuit.")
        simulator = qiskit.Aer.get_backend('aer_simulator')
        circ_q = self.to_qiskit_circuit()
        result = simulator.run(circ_q).result()
        return result.get_counts(circ_q)

    # print without qiskit
    def __str__(self) -> str:
        circuit_string = ""
        for i in range(self.nr_qubits):
            if i != 0:
                circuit_string += "\n"
            circuit_string += "q" + str(i) + " --"
            for gate in self.gates:

                if gate.target() == i:
                    circuit_string += str(gate)
                elif gate.nr_qubits() > 1 and i in gate.control():
                    circuit_string += pad_to_length("@", NAME_LENGTH)
                else:
                    circuit_string += pad_to_length("", NAME_LENGTH)
                circuit_string += "-"
        return circuit_string

    def __out_of_range_check(self, qubit):
        if qubit >= self.nr_qubits:
            raise Exception(
                f"Index '{qubit}' out of range for size '{self.nr_qubits}'.")
