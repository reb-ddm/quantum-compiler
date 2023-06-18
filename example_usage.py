import compiler

# create a circuit with 3 qubits
circuit = compiler.Circuit(3)

# add some gates
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.cnot(0, 1)
circuit.cnot(1, 2)
circuit.cnot(2, 0)
circuit.measure([0, 1, 2])

# visualize using qiskit
print(circuit.qiskit_draw())

# compile for an IBM 5 qubit architecture
ibmq_manila_coupling_map = [[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3]]

circuit.compile(ibmq_manila_coupling_map, 5)

# visualize compiled circuit
print(circuit.qiskit_draw())

