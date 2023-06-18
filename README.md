# Quantum Compiling

A simple compiler for quantum computers implemented for the seminar "Advanced Topics in Quantum Computing" at TUM.

## Description

The compiler implements functions to represent a quantum circuit, decompose it to the set of gates CX, X, SX, and $R_z(\phi)$, route it to arbitrary quantum hardware and simulate the circuit using qiskit.

See `compiler_demonstration.ipynb` for an example of the functions of the compiler.

The file `decomposition.ipynb` describes how the decomposition is implemented.

The file `test_main.py` runs all the tests defined in the folder `test`.

## Usage

```python
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
```

## Dependencies

- numpy
- qiskit
