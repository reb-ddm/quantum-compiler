# Quantum Compiling

A simple compiler for quantum computers implemented for the seminar "Advanced Topics in Quantum Computing" at TUM.

## Description

The compiler implements functions to represent a quantum circuit, decompose it to the set of gates CX, X, SX, and $R_z(\phi)$, route it to arbitrary quantum hardware and simulate the circuit using qiskit.

See `compiler_demonstration.ipynb` for an example of the functions of the compiler.

The file `decomposition.ipynb` describes how the decomposition is implemented.

The file `test_main.py` runs all the tests defined in the folder `test`.

## Dependencies

- numpy
- qiskit
