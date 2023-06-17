"""
Defines the class 'Gate', which models a quantum gate. 
The gates can be decomposed to the set of native gates ['cx', 'id', 'rz', 'x', 'sx'],
which is the set of native gates for most of the IBM quantum computers 
that can be accessed on the cloud.
"""
import copy
import numpy as np
import qiskit

EPS = 1e-7

NAME_LENGTH = 6  # used for printing the circuit in ASCII
native_gates = ['cx', 'id', 'rz', 'x', 'sx']
supported_gates = ['u', 'x', 'y', 'z', 'h', 'rz', 'ry', 'rx', 's', 'sdg', 't',
                   'tdg', 'sx', 'sxdg', 'ssx', 'id', 'cy', 'cz', 'swap', 'cx', 'crx', 'cu',
                   'cs', 'ct',
                   'measure', 'reset']


def pad_to_length(name_str, string_length):
    padding_front = max(0, (string_length - len(name_str)) // 2)
    padding_end = max(0, (string_length - len(name_str)) //
                      2 + (string_length - len(name_str)) % 2)
    return '-' * padding_front + name_str + '-' * padding_end


def zy_decomposition(theta, phi, lam, rho):
    """
    Returns angles α, β, γ, δ where `U = e^iα*Rz(β)*Ry(γ)*Rz(δ)`.

    See `decomposition.ipynb` for the definition of U.
    """
    delta = lam
    beta = phi
    gamma = theta
    alpha = beta/2 + delta/2 + rho

    return alpha, beta, gamma, delta


class Gate:
    def __init__(self, name, qubits, params=[]):
        self.name = name
        # the first element of the qubits list is the target, all the following qubits are control qubits
        if isinstance(qubits, list):
            # check for duplicates
            if len(qubits) != len(set(qubits)):
                raise Exception(
                    f"A qubit can't be target and control at the same time, or used as control twice.")
            self.qubits = qubits
        else:
            self.qubits = [qubits]
        # the gates 'cy' and 'cz' and 'swap' and 'cu' and 'crx' have exactly 1 control qubit
        # the gates 'cx' and 'ssx' are allowed an arbitrary number of control qubits, but 'cx' needs at least one
        # all other gates are not allowed to have any control qubits
        if name not in supported_gates:
            raise Exception(f"Unknown gate: '{name}'")
        elif name == 'cy' or name == 'cz' or name == 'swap' or name == 'cu' or name == 'crx' or name == 'cs' or name == 'ct':
            if len(self.qubits) != 2:
                raise Exception(
                    f"The gate '{name}' must have exactly 1 control qubit. {len(self.qubits) - 1} given.")
        elif name == 'ssx':
            pass
        elif name == 'cx':
            if len(self.qubits) < 2:
                raise Exception(
                    f"The gate '{name}' must have at least 1 control qubit. {len(self.qubits) - 1} given.")
        elif len(self.qubits) != 1:
            raise Exception(
                f"The gate '{name}' must have exactly 1 qubit. {len(self.qubits)} given.")
        # check right number of parameters
        if (name == 'u' or name == 'cu') and len(params) != 4:
            raise Exception(
                f"The gate '{name}' must have exactly 4 parameters. {len(params)} given.")
        elif (name == 'rz' or name == 'ry' or name == 'rx' or name == 'crx' or name == 'ssx') and len(params) != 1:
            raise Exception(
                f"The gate '{name}' must have exactly one parameter. {len(params)} given.")
        elif name not in ['u', 'cu', 'rz', 'ry', 'rx', 'crx', 'ssx'] and len(params) != 0:
            raise Exception(
                f"The gate '{name}' must have exactly zero parameters. {len(params)} given.")
        self.params = params

    def target(self):
        """Returns the target qubit."""
        return self.qubits[0]

    def control(self):
        """Returns the control qubits as a list."""
        return self.qubits[1:]

    def nr_qubits(self):
        """Returns how many qubits are part of the gate."""
        return len(self.qubits)

    # functions used for decomposition
    def decompose(self):
        gates = [self]
        decomposed_gates = []
        # decompose to 2-qubit gates
        for gate in gates:
            decomposed_gates += gate._decompose_multiple_qubit_gates()
        gates = decomposed_gates
        decomposed_gates = []
        # decompose to CNOT and U
        for gate in gates:
            decomposed_gates += gate._decompose_to_cx_rz_u()
        gates = decomposed_gates
        decomposed_gates = []
        # decompose U to ['cx', 'id', 'rz', 'x', 'sx']
        for gate in gates:
            decomposed_gates += gate._decompose_u_gate()
        return decomposed_gates

    def _decompose_multiple_qubit_gates(self):
        """
        Translates gates with more than one control line to a circuit with only single qubit
        gates and two qubit gates.

        For the decomposition of Toffoli gates no ancilla qubits are used, 
        thus it is not the most efficient decomposition (O(n^2) gates for n control lines).

        Only Toffoli gate decomposition is supported for now.

        Decomposition described in the paper "Elementary gates for quantum computation"
        by Barenco et al. (Lemma 7.5). A description of my method can also be found in `decomposition.ipynb`.
        """
        if self.nr_qubits() <= 2:
            return [self]
        elif self.name == 'cx' or self.name == 'ssx':
            control = self.control()
            target = self.target()
            first_control = control.pop(0)
            if self.name == 'cx':
                k = 0
            else:
                k = self.params[0]
            if k >= 0:
                k_succ = k + 1
            else:
                k_succ = k - 1
            first_gate = [Gate('ssx', [target, first_control], [k_succ])]
            second_gate = Gate('cx', [first_control] +
                               control)._decompose_multiple_qubit_gates()
            third_gate = [Gate('ssx', [target, first_control], [-k_succ])]

            fifth_gate = Gate('ssx', [target] + control,
                              [k_succ])._decompose_multiple_qubit_gates()
            return first_gate + second_gate + third_gate + copy.deepcopy(second_gate) + fifth_gate
        else:
            return [self]

    def _decompose_to_cx_rz_u(self):
        """
        Translates one and two qubit gates to CX, RZ and U gates, except if they are
        already in the set of native gates ['cx', 'rz', 'x', 'sx'].
        The multiple qubit gates remain unchanged.
        The phase of some gates is ignored.

        The correctness of the decomposition can be checked by matrix multiplication.
        """
        # single qubit gates
        if self.name == 'y':
            return [Gate('u', self.qubits, params=[
                np.pi, np.pi/2, np.pi/2, 0])]
        elif self.name == 'z':
            return [Gate('u', self.qubits, params=[0, 0, np.pi, 0])]
        elif self.name == 'h':
            return [
                Gate('u', self.qubits, params=[np.pi/2, 0, np.pi, 0])]
        elif self.name == 'ry':
            return self._decompose_ry_gate()
        elif self.name == 'rx':
            return [Gate('u', self.qubits, params=[self.params[0], -np.pi/2, np.pi/2, 0])]
        elif self.name == 's':
            return [Gate('u', self.qubits, params=[0, 0, np.pi/2, 0])]
        elif self.name == 'sdg':
            return [Gate('u', self.qubits, params=[0, 0, -np.pi/2, 0])]
        elif self.name == 't':
            return [Gate('u', self.qubits, params=[0, 0, np.pi/4, 0])]
        elif self.name == 'tdg':
            return [Gate('u', self.qubits, params=[0, 0, -np.pi/4, 0])]
        elif self.name == 'sxdg':
            return [
                Gate('u', self.qubits, params=[-np.pi/2, -np.pi/2, np.pi/2, -np.pi/4])]
        elif self.name == 'id':
            return []
        elif self.name == 'ssx':
            return self._decompose_ssx_to_cu()._decompose_cu_gate()
        # two qubit gates
        elif self.name == 'cy':
            return [Gate('rz', self.target(), params=[-np.pi/2]), Gate('cx', self.qubits), Gate('rz', self.target(), params=[np.pi/2])]
        elif self.name == 'cz':
            return [Gate('u', self.target(), params=[np.pi/2, 0, np.pi, 0]), Gate('cx', self.qubits),
                    Gate('u', self.target(), params=[np.pi/2, 0, np.pi, 0])]
        elif self.name == 'swap':
            return [Gate('cx', self.qubits), Gate('cx', [self.qubits[1], self.qubits[0]]), Gate('cx', self.qubits)]
        elif self.name == 'crx' or self.name == 'cu' or self.name == 'cs' or self.name == 'ct':
            return self._decompose_cu_gate()
        else:
            return [self]

    def _decompose_ssx_to_cu(self):
        if self.name == 'ssx':
            k = self.params[0]
            if k == 0:
                if self.nr_qubits() == 1:
                    return Gate('x', self.qubits)
                else:
                    return Gate('cx', self.qubits)
            elif k == 1:
                if self.nr_qubits() == 1:
                    return Gate('sx', self.qubits)
                else:
                    return Gate('cu', self.qubits, params=[np.pi/2**k, -np.pi/2, np.pi/2, np.pi/2**(k+1)])
            if self.nr_qubits() == 1:
                gate_name = 'u'
            else:
                gate_name = 'cu'
            if k >= 0:
                return Gate(gate_name, self.qubits, params=[np.pi/2**k, -np.pi/2, np.pi/2, np.pi/2**(k+1)])
            else:
                k = np.abs(k)
                return Gate(gate_name, self.qubits, params=[-np.pi/2**k, -np.pi/2, np.pi/2, -np.pi/2**(k+1)])

    def _decompose_cu_gate(self):
        """
        Translates CRX and CU gates with one control line to U and CNOT gates.
        Other gates remain unchanged.

        Decomposition described in the paper "Elementary gates for quantum computation"
        by Barenco et al. on (Lemma 4.3, Lemma 5.1).
        """
        decomposed_gates = []
        if self.name in ['crx', 'cu', 'ssx', 'cs', 'ct'] and self.nr_qubits() == 2:
            if self.name == 'crx':
                theta, phi, lam, rho = self.params[0], -np.pi/2, np.pi/2, 0
            elif self.name == 'ssx':
                k = self.params[0]
                if k == 0:
                    return [Gate('cx', self.qubits)]
                elif k > 0:
                    theta, phi, lam, rho = np.pi/2**k, - \
                        np.pi/2, np.pi/2, np.pi/2**(k+1)
                else:
                    k = np.abs(k)
                    theta, phi, lam, rho = -np.pi/2**k, - \
                        np.pi/2, np.pi/2, -np.pi/2**(k+1)
            elif self.name == 'cs':
                theta, phi, lam, rho = 0, 0, np.pi/2, 0
            elif self.name == 'ct':
                theta, phi, lam, rho = 0, 0, np.pi/4, 0
            else:  # CU gate
                theta, phi, lam, rho = self.params[0], self.params[1], self.params[2], self.params[3]
            alpha, beta, gamma, delta = zy_decomposition(theta, phi, lam, rho)
            # phase shift described in Lemma 5.2 and Corollary 5.3
            if alpha != 0:
                phase_shift = [
                    Gate('u', self.control(), params=[0, 0, alpha, 0])]
            else:
                phase_shift = []
            A = Gate('ry', self.target(), params=[
                     gamma/2])._decompose_ry_gate() + [Gate('rz', self.target(), params=[beta])]
            B = [Gate('rz', self.target(), params=[-(beta + delta)/2])] + \
                Gate('ry', self.target(),
                     params=[-gamma/2])._decompose_ry_gate()
            C = [Gate('rz', self.target(), params=[(delta - beta)/2])]
            CX = [Gate('cx', self.qubits)]
            decomposed_gates = C + CX + B + copy.deepcopy(CX) + A + phase_shift
            # in the paper they put A - B - C in this order, but it should actually be C - B - A
            # because A*B*C = I and not the other way around
        else:
            decomposed_gates = [self]
        return decomposed_gates

    def _decompose_u_gate(self):
        """Decomposes U gates using the ZY decomposition."""
        decomposed_gates = []
        if self.name == 'u':
            # we ignore the phase alpha
            _, beta, gamma, delta = zy_decomposition(
                self.params[0], self.params[1], self.params[2], self.params[3])
            ry_decomposed = Gate('ry', self.qubits, params=[
                                 gamma])._decompose_ry_gate()
            decomposed_gates = []
            if delta != 0:
                decomposed_gates += [Gate('rz', self.qubits, params=[delta])]
            decomposed_gates += ry_decomposed
            if beta != 0:
                decomposed_gates += [Gate('rz', self.qubits, params=[beta])]
        else:
            decomposed_gates = [self]
        return decomposed_gates

    def _decompose_ry_gate(self):
        """
        Translates RY to RZ and SX gates.
        The phase is ignored.
        Other gates remain unchanged.

        A description of the decomposition can be found in `decomposition.ipynb`.
        """
        decomposed_gates = []
        if self.name == 'ry':
            decomposed_gates = [
                Gate('rz', self.qubits, params=[-np.pi]), Gate('sx', self.qubits, params=[])]
            if np.pi != self.params[0]:
                decomposed_gates += [Gate('rz', self.qubits,
                                          params=[np.pi - self.params[0]])]
            decomposed_gates += [Gate('sx', self.qubits, params=[])]
        else:
            decomposed_gates = [self]
        return decomposed_gates

    # functions used for optimization of quantum circuits
    def is_inverse(self, other):
        """
        If it returns True, then gate1 and gate2 are inverse.
        (Not necessarily the other way around, for example it returns False 
        if any of the gates is a U gate)
        """
        if self.target() != other.target() or set(self.control()) != set(other.control()):
            return False

        gate1 = self.name
        gate2 = other.name

        if gate1 == gate2 + 'dg' or gate2 == gate1 + 'dg':
            return True
        self_inverse_gates = ['x', 'y', 'z', 'h', 'cx', 'cy', 'cz', 'swap']
        if gate1 == gate2 and gate2 in self_inverse_gates:
            return True

        if gate1 == 'rz' and gate2 == 'rz':
            return (self.params[0] + other.params[0]) % (2 * np.pi) == 0

        if gate1 == gate2 and gate1 == 'ssx' and self.params[0] == -other.params[0]:
            return True
        if gate1 == gate2 and gate1 in ['rx', 'ry', 'rz', 'crx'] and (np.abs((self.params[0] + other.params[0]) - 2*np.pi) < EPS
                                                                      or np.abs((self.params[0] + other.params[0])) < EPS or np.abs((self.params[0] + other.params[0]) + 2*np.pi) < EPS):
            return True
        return False

    # functions used for routing
    def map_qubits(self, mapping):
        mapped_gate = copy.copy(self)
        mapped_gate.qubits = [mapping[i] for i in self.qubits]
        return mapped_gate

    # other useful functions
    def to_qiskit_gate(self):
        """Converts the Gate to a qiskit.Gate."""
        # single qubit gates
        if self.name == 'u':
            return qiskit.circuit.library.UGate(self.params[0], self.params[1], self.params[2])
        elif self.name == 'x':
            return qiskit.circuit.library.XGate()
        elif self.name == 'y':
            return qiskit.circuit.library.YGate()
        elif self.name == 'z':
            return qiskit.circuit.library.ZGate()
        elif self.name == 'h':
            return qiskit.circuit.library.HGate()
        elif self.name == 'rz':
            return qiskit.circuit.library.RZGate(self.params[0])
        elif self.name == 'ry':
            return qiskit.circuit.library.RYGate(self.params[0])
        elif self.name == 'rx':
            return qiskit.circuit.library.RXGate(self.params[0])
        elif self.name == 's':
            return qiskit.circuit.library.SGate()
        elif self.name == 'sdg':
            return qiskit.circuit.library.SdgGate()
        elif self.name == 't':
            return qiskit.circuit.library.TGate()
        elif self.name == 'tdg':
            return qiskit.circuit.library.TdgGate()
        elif self.name == 'sx':
            return qiskit.circuit.library.SXGate()
        elif self.name == 'sxdg':
            return qiskit.circuit.library.SXdgGate()
        elif self.name == 'id':
            return qiskit.circuit.library.IGate()

        # two qubit gates
        elif self.name == 'cy':
            return qiskit.circuit.library.CYGate()
        elif self.name == 'cz':
            return qiskit.circuit.library.CZGate()
        elif self.name == 'cs':
            return qiskit.circuit.library.SGate().control(1)
        elif self.name == 'ct':
            return qiskit.circuit.library.TGate().control(1)
        elif self.name == 'cu':
            return qiskit.circuit.library.CUGate(self.params[0], self.params[1], self.params[2], self.params[3])
        elif self.name == 'crx':
            return qiskit.circuit.library.CRXGate(self.params[0])
        elif self.name == 'swap':
            return qiskit.circuit.library.SwapGate()

        # arbitrary qubit gates
        elif self.name == 'cx':
            return qiskit.circuit.library.MCXGate(num_ctrl_qubits=len(self.control()))
        elif self.name == 'ssx':
            return self._decompose_ssx_to_cu().to_qiskit_gate()

        # measure and reset
        elif self.name == 'measure':
            return qiskit.circuit.library.Measure()
        elif self.name == 'reset':
            return qiskit.circuit.library.Reset()

        else:
            return qiskit.circuit.Gate(name=self.name, num_qubits=self.nr_qubits(), params=self.params)

    def __str__(self) -> str:
        name_str = self.name.upper()
        if self.name == 'rz':
            name_str = 'R' + str(round(self.params[0], 2))
        return pad_to_length(name_str, NAME_LENGTH)
