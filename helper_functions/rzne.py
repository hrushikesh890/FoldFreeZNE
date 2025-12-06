"""RZNE (Reliability Zero Noise Extrapolation) helper functions.

This module provides functions for noise simulation, extrapolation, and
error mitigation in quantum circuits.
"""

import os
import logging
import warnings
import collections

import numpy as np
import scipy
import supermarq

from qiskit import Aer, execute, IBMQ
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.providers.aer.noise import NoiseModel

warnings.simplefilter("ignore", np.ComplexWarning)


# function to get ESP
def get_esp(circuit, noise_model, backend):
    backend_prop=backend.properties()
    cx_reliability={}
    rz_reliability={}
    sx_reliability={}
    x_reliability={}
    readout_reliability={}
    cx_num={}
    for ginfo in backend_prop.gates:
        if ginfo.gate=="cx":
            for param in ginfo.parameters:
                if param.name=="gate_error":
                    g_reliab = 1.0 - param.value
                    break
            cx_reliability[(ginfo.qubits[0], ginfo.qubits[1])] = g_reliab
            cx_num[(ginfo.qubits[0], ginfo.qubits[1])] = 0
        if ginfo.gate=="rz":
            for param in ginfo.parameters:
                if param.name=="gate_error":
                    g_reliab = 1.0 - param.value
                    break
            rz_reliability[(ginfo.qubits[0])] = g_reliab
        if ginfo.gate=="sx":
            for param in ginfo.parameters:
                if param.name=="gate_error":
                    g_reliab = 1.0 - param.value
                    break
            sx_reliability[(ginfo.qubits[0])] = g_reliab
        if ginfo.gate=="x":
            for param in ginfo.parameters:
                if param.name=="gate_error":
                    g_reliab = 1.0 - param.value
                    break
            x_reliability[(ginfo.qubits[0])] = g_reliab
    for i in range(backend.configuration().n_qubits):
        readout_reliability[(i)]=1.0-backend_prop.readout_error(i)
    qc = transpile(circuit, basis_gates= noise_model.basis_gates,
                      coupling_map=backend.configuration().coupling_map,
                      seed_transpiler =0,
                      optimization_level=0)
    dag = circuit_to_dag(qc)
    esp=1
    for node in dag.op_nodes():
        if node.name == "rz":
            key=node.qargs[0]._index
            esp=esp*rz_reliability[key]
        elif node.name == "sx":
            key=node.qargs[0]._index
            esp=esp*sx_reliability[key]
        elif node.name == "x":
            key=node.qargs[0]._index
            esp=esp*x_reliability[key]
        elif node.name == "cx":
            key=(node.qargs[0]._index, node.qargs[1]._index)
            esp=esp*cx_reliability[key]
        elif node.name == "measure":
            key=node.qargs[0]._index
            esp=esp*readout_reliability[key]
    return esp

# Noisy simulation 
# In: circuit: QuantumCircuit
# In: s: num of shots (default: 32768)
# Out: dictionary of counts of each bitstring
def noise_simulation(circuit, s = 32768):
    circuit.remove_final_measurements()
    circuit.measure_all()
    circuit = transpile(circuit, backend = backend, optimization_level = 0, scheduling_method = 'asap')
    result = execute(circuit, Aer.get_backend('qasm_simulator'),
                 coupling_map=coupling_map,
                 basis_gates=basis_gates,
                 noise_model=noise_model, shots = s, simulator_seed = 10, optimization_level=0).result()
    counts1 = result.get_counts(circuit)
    circuit.remove_final_measurements()
    counts1 = renormalize(counts1)
    return collections.Counter(counts1)

# Method for decimal to binary
def binary(num, pre='0b', length=8, spacer=0):
    return '{0}{{:{1}>{2}}}'.format(pre, spacer, length).format(bin(num)[2:])

# Method to perform ideal simulation
# In: circuit: QuantumCircuit
# In: s: num of shots (default: 32768)
# Out: dictionary of counts of each bitstring
def ideal_simulation(circuit, s = 32768):
    circuit.remove_final_measurements()
    circuit.measure_all()
    result = execute(circuit, Aer.get_backend('qasm_simulator'), shots = s, simulator_seed = 10).result()
    counts1 = result.get_counts(circuit)
    circuit.remove_final_measurements()
    return collections.Counter(counts1)

# Method to renormalize the vector
# In ip_vector: Vector to be normalized
# Out normalized vector
def renormalize(ip_vector):
    s = 0
    for key in ip_vector.keys():
        s += ip_vector[key]
    
    for key in ip_vector.keys():
        ip_vector[key] = (ip_vector[key]/s)
    
    return ip_vector

# Method to perform RZNE for each bitstring
# In ip_vector: Input vector
# In esp: ESP of the circuit
# In profiled_ip: Profiled vector, can either be profiled or obtained via extrapolation to infinite noise case
# Out dictionary of mitigated counts of each bitstring
def extrapolate_each_vec(ip_vector, esp, profiled_ip):
    op_dict = {}
    for key in ip_vector.keys():
        data = np.array([[1-esp, ip_vector[key]], [1, profiled_ip[key]]])
        fit = np.polyfit(data[:,0], data[:,1] ,1)
        line = np.poly1d(fit)
        op_dict[key] = (line([0])[0])
        if (op_dict[key] < 0):
            op_dict[key] = 0
    #print(ip_vector)
    op_dict = renormalize(op_dict)
    return collections.Counter(op_dict)

# Method to get profiled state.
# In    nqbs: Number of qubits in the circuit
# Out   profiled vector
def get_profiled_state(nqbs, tt = 2):
    vqe = supermarq.hamiltonian_simulation.HamiltonianSimulation(nqbs, 1, 2)
    vqe_circuit = vqe.circuit()
    circuit = supermarq.converters.cirq_to_qiskit(vqe_circuit)
    circuit.remove_final_measurements()
    for i in range(0, 200):
        for j in range (0, nqbs):
            circuit.x(j)
            circuit.x(j)
        for j in range (1, nqbs):
            circuit.cx(j-1, j)
            circuit.cx(j-1, j)
    profiled_ip = noise_simulation(circuit)
    return profiled_ip, circuit

# Method to fold gates
# In    circuit: QuantumCircuit
# In    layers: Num of layers to be folded
# Out   folded QuantumCircuit
def fold_gates(circuit, layers):
    circ = circuit.copy()
    circ.remove_final_measurements()
    for i in range (0, layers):
        for k in range(1, circ.num_qubits):
            circ.cx(k-1, k)
            circ.cx(k-1, k)
    return circ

# Method for exponentail extrapolation
# y(k, b, m) = 1 - k*e^(-m*x/t) + b
def exponential2(x, k, b, m):
    return 1-k*np.exp(-m*x/140)+b

# Method for exponentail extrapolation
# y(k, b) = 1 - k*e^(-x/t) + b
def exponential(x, k, b):
    return 1-k*np.exp(-x/140)+b

# Method for exponential tRZNE 
# In    ip_vector: Input vector to be mitigated
# In    profile_ip: Profiled vector 
# In    t1, t2: circuit times of original circuit and the infinite noise case
# Out   mitigated statevector
def extrap_everything_time(ip_vector, profile_ip, t2, t1):
    op_dict = {}
    for key in ip_vector.keys():
        popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, [t2, t1], [profile_ip[key], ip_vector[key]], p0=[-1, 1], maxfev = 10000)
        op_dict[key] = exponential(0, popt_exponential[0], popt_exponential[1])
        if (op_dict[key] < 0):
            op_dict[key] = 0
    op_dict = renormalize(op_dict)
    #print(ip_vector, op_dict)
    return collections.Counter(op_dict)

# Method to get the circuit time of a given circuit
# In    circuit: Input circuit
# In    backend: backend machine
# In    lvl: Optimization level (default 0)
# Out   circuit time in us
def get_time(circuit, backend, lvl = 0):
    circuit = transpile(circuit, backend = backend, optimization_level = lvl, scheduling_method = 'asap')
    max_time = 0
    for i in range(0, num_qubits):
        t = circuit.qubit_duration(i)
        
        if (t > max_time):
            max_time = t
    #print(lvl, max_time)        
    return max_time/1000

# Method to get infinite noise state (decoherence noise) or profiled input.
# In    nqbs: Number of qubits in the circuit
# In    backend: Qiskit backend object of the machine where the computation has to be performed
# In    true_prof: Variable to give profiled or infinite noise case. True for profiled and false for inifinite noise. (Default: True)
# Out   profiled vector, and circuit time
def get_profiled_state_decoh(nqbs, tt, backend, true_prof = True):
    if true_prof:
        vqe = supermarq.hamiltonian_simulation.HamiltonianSimulation(nqbs, 1, tt)
        vqe_circuit = vqe.circuit()
        circuit = supermarq.converters.cirq_to_qiskit(vqe_circuit)
        circuit.remove_final_measurements()
        for i in range(0, 500):
            for j in range (0, nqbs):
                circuit.x(j)
                circuit.x(j)
            for j in range (1, nqbs):
                circuit.cx(j-1, j)
                circuit.cx(j-1, j)
        profiled_ip = noise_simulation(circuit)
        profiled_ip = renormalize(profiled_ip)
        time = get_time(circuit, backend, 0)
    else:
        time = 14000
        getbin = lambda x, n: format(x, 'b').zfill(n)
        profiled_ip = {}
        for i in range(0, 2**nqbs):
            if i != 0:
                profiled_ip[getbin(i, nqbs)] = 0
            else:
                profiled_ip[getbin(i, nqbs)] = 1 
        profiled_ip = collections.Counter(profiled_ip)
    return profiled_ip, time

# Method to perform exponential extrapolation on each element of the vector.
# In    circuit: QuantumCiruit: Original circuit whose noise has to be mitigated 
# In    profile_ip: Profiled or inifinte noise state vector
# In    t_prof: time ptofiled.
# Out   mitigated statevector
def extrap_everything_decoh(circuit, profile_ip, t_prof):
    op_dict = {}
    circ1 = [profile_ip, noise_simulation(circuit)]
    t = [t_prof, get_time(circuit, backend, 0)]
    for k in range(1, 4):
        circ1.append(noise_simulation(fold_gates(circuit, k)))
        t.append(get_time(fold_gates(circuit, k), backend, 0))
    
    for key in circ1[1].keys():
        c = []
        for z in range(0, len(circ1)):
            c.append(circ1[z][key])
        if (key in profile_ip.keys()):
            #do nothing
            zz = 1
        else: 
            profile_ip[key] = 0
        getbin = lambda x, n: format(x, 'b').zfill(n)   
        if (getbin(0, circuit.num_qubits) == key):
            print(c, t)
        popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, t, c, p0=[-1, 1], maxfev = 10000)
        op_dict[key] = exponential(0, popt_exponential[0], popt_exponential[1])
        if (op_dict[key] < 0):
            op_dict[key] = 0
    op_dict = renormalize(op_dict)
    #print(ip_vector, op_dict)
    return collections.Counter(op_dict)

# Method to get infinite noise state (dpolarizing noise) or profiled input.
# In    nqbs: Number of qubits in the circuit
# In    backend: Qiskit backend object of the machine where the computation has to be performed
# In    true_prof: Variable to give profiled or infinite noise case. True for profiled and false for inifinite noise. (Default: True)
# Out   profiled vector
def get_profiled_state_depol(nqbs, tt, backend, true_prof = True):
    if true_prof:
        vqe = supermarq.hamiltonian_simulation.HamiltonianSimulation(nqbs, 1, tt)
        vqe_circuit = vqe.circuit()
        circuit = supermarq.converters.cirq_to_qiskit(vqe_circuit)
        circuit.remove_final_measurements()
        for i in range(0, 500):
            for j in range (0, nqbs):
                circuit.x(j)
                circuit.x(j)
            for j in range (1, nqbs):
                circuit.cx(j-1, j)
                circuit.cx(j-1, j)
        profiled_ip = noise_simulation(circuit)
        profiled_ip = renormalize(profiled_ip)
    else:
        getbin = lambda x, n: format(x, 'b').zfill(n)
        profiled_ip = {}
        for i in range(0, 2**nqbs):
            if i != 0:
                profiled_ip[getbin(i, nqbs)] = 1/(2**nqbs)
            else:
                profiled_ip[getbin(i, nqbs)] = 1/(2**nqbs)
        profiled_ip = collections.Counter(profiled_ip)
    return profiled_ip

# Method to perform RZNE on each element of the vector.
# In    ip_vector: Input vector 
# In    profiled_ip: Profiled or inifinte noise state vector
# In    esp: ESP of the original circuit.
# Out   mitigated statevector
def extrap_everything_depol(ip_vector, esp, profiled_ip):
    op_dict = {}
    for key in ip_vector.keys():
        data = np.array([[1-esp, ip_vector[key]], [1, profiled_ip[key]]])
        fit = np.polyfit(data[:,0], data[:,1] ,1)
        line = np.poly1d(fit)
        op_dict[key] = (line([0])[0])
        if (op_dict[key] < 0):
            op_dict[key] = 0
    #print(ip_vector)
    op_dict = renormalize(op_dict)
    return collections.Counter(op_dict)