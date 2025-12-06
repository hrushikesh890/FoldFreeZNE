# FoldFreeZNE: Fold-Free Zero Noise Extrapolation

This repository contains the implementation of RZNE (Reliability Zero Noise Extrapolation) and Cut ZNE techniques for quantum error mitigation.

## Reference

Please refer to our paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/10313621)

## Installation

Install the required dependencies:

```bash
pip install qiskit qiskit-aer mitiq supermarq mthree scipy numpy
```

## Configuration

### IBMQ Account Setup

For security, set your IBMQ token as an environment variable:

```bash
export IBMQ_TOKEN="your_token_here"
```

Alternatively, the code will attempt to load a previously saved account.

## Usage

### Running Benchmarks

Execute the main runner script:

```bash
python runner.py
```

This will run various quantum circuit benchmarks (Hamiltonian Simulation, VQE, QAOA) and save results to CSV files.

### Main Modules

- `rzne.py`: Main RZNE implementation with noise simulation and extrapolation functions
- `cutrzne.py`: Cut ZNE implementation (cutqc-related code)
- `runner.py`: Script for running benchmark experiments
- `helper_functions/`: Utility functions for benchmarks, metrics, and conversions

## Code Structure

- **RZNE Functions**: Noise simulation, ESP calculation, extrapolation methods
- **Benchmark Execution**: Functions for running various quantum circuit benchmarks
- **Error Mitigation**: Integration with Mitiq for ZNE, CDR, and other techniques

## Notes

- The `cutqc` related code is excluded from cleanup as requested
- MATLAB integration is optional and can be commented out if not needed
