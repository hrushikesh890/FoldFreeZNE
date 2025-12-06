"""RZNE (Reliability Zero Noise Extrapolation) main module.

This module provides functions for noise simulation, extrapolation, and
error mitigation in quantum circuits using RZNE techniques.
"""

import os
import glob
import json
import logging
import copy
import collections
import pickle
import warnings
from datetime import date

import numpy as np
import scipy
import qiskit
import supermarq
import mthree

from mitiq import cdr, ddd, zne
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne.inference import LinearFactory
from qiskit import IBMQ, Aer, execute, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info.analysis import hellinger_fidelity

# MATLAB engine (optional, comment out if not using MATLAB)
# import matlab.engine
# import matlab
# eng = matlab.engine.start_matlab()

warnings.simplefilter("ignore", np.ComplexWarning)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# Comment this line if using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Date string for output directories
today = date.today().strftime("%Y-%m-%d")


# IBMQ account configuration
# Note: Set IBMQ_TOKEN environment variable for security
# IBMQ.delete_account()
if 'IBMQ_TOKEN' in os.environ:
    IBMQ.save_account(token=os.environ['IBMQ_TOKEN'])
    provider = IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-ncsu', group='nc-state', project='quantum-error-mo')
    backend = provider.get_backend('ibmq_guadalupe')
else:
    # Fallback: load existing saved account
    provider = IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-ncsu', group='nc-state', project='quantum-error-mo')
    backend = provider.get_backend('ibmq_guadalupe')
# Backend configuration
noise_model = NoiseModel.from_backend(
    backend, readout_error=False, thermal_relaxation=True, gate_error=True
)
machine_list = ['ibmq_guadalupe']
coupling_map = backend.configuration().coupling_map
backend2 = QasmSimulator(noise_model=noise_model)
basis_gates = noise_model.basis_gates

# Global constants
num_qubits = 16
mit = mthree.M3Mitigation(backend2)
mit.cals_from_system(range(16))
current_test = "rand3"
t1_time = 79.29

def get_state_list(nqbs):
    """Generate list of all possible bitstrings for n qubits.
    
    Args:
        nqbs: Number of qubits
        
    Returns:
        List of binary strings representing all possible states
    """
    getbin = lambda x: format(x, 'b').zfill(nqbs)
    return [getbin(i) for i in range(2**nqbs)]

# Constants
M = 32768  # Default number of shots
S = {
    3: get_state_list(3),
    4: ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111',
        '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111'],
    5: ['00000', '00001', '00010', '00011', '00100', '00101', '00110', '00111',
        '01000', '01001', '01010', '01011', '01100', '01101', '01110', '01111',
        '10000', '10001', '10010', '10011', '10100', '10101', '10110', '10111',
        '11000', '11001', '11010', '11011', '11100', '11101', '11110', '11111'],
    6: get_state_list(6),
    7: get_state_list(7),
    8: get_state_list(8),
    12: get_state_list(12),
    16: get_state_list(16)
}

C = ['ibmq_guadalupe']

def get_esp(circuit, noise_model, backend):
    """Calculate Expected Success Probability (ESP) for a circuit.
    
    Args:
        circuit: QuantumCircuit to analyze
        noise_model: NoiseModel for the backend
        backend: Backend configuration
        
    Returns:
        ESP value (float)
    """
    backend_prop = backend.properties()
    cx_reliability = {}
    rz_reliability = {}
    sx_reliability = {}
    x_reliability = {}
    readout_reliability = {}
    cx_num = {}
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
    for i in range(num_qubits):
        readout_reliability[i] = 1.0 - backend_prop.readout_error(i)
    
    qc = transpile(
        circuit,
        basis_gates=noise_model.basis_gates,
        coupling_map=backend.configuration().coupling_map,
        seed_transpiler=0,
        optimization_level=0
    )
    dag = circuit_to_dag(qc)
    esp = 1
    for node in dag.op_nodes():
        if node.name == "rz":
            key = node.qargs[0]._index
            esp *= rz_reliability[key]
        elif node.name == "sx":
            key = node.qargs[0]._index
            esp *= sx_reliability[key]
        elif node.name == "x":
            key = node.qargs[0]._index
            esp *= x_reliability[key]
        elif node.name == "cx":
            key = (node.qargs[0]._index, node.qargs[1]._index)
            esp *= cx_reliability[key]
        elif node.name == "measure":
            key = node.qargs[0]._index
            esp *= readout_reliability[key]
    return esp

def noise_simulation(circuit, s=32768, readout_mitigated=False):
    """Simulate a circuit with noise.
    
    Args:
        circuit: QuantumCircuit to simulate
        s: Number of shots (default: 32768)
        readout_mitigated: Whether to apply readout error mitigation
        
    Returns:
        Counter of measurement results
    """
    qnum = circuit.num_qubits
    circuit.remove_final_measurements()
    circuit.measure_all()
    
    circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=0,
        scheduling_method='asap'
    )
    result = execute(
        circuit,
        Aer.get_backend('qasm_simulator'),
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        noise_model=noise_model,
        shots=s,
        simulator_seed=10,
        optimization_level=0
    ).result()
    counts1 = result.get_counts(circuit)
    
    if readout_mitigated:
        quasis = mit.apply_correction(counts1, range(qnum))
        counts1 = quasis.nearest_probability_distribution()
        print("Mitigated")
    
    circuit.remove_final_measurements()
    counts1 = renormalize(counts1)
    return collections.Counter(counts1)

def binary(num, pre='0b', length=8, spacer=0):
    """Convert number to binary string with formatting.
    
    Args:
        num: Number to convert
        pre: Prefix for binary string
        length: Minimum length of output
        spacer: Spacer character
        
    Returns:
        Formatted binary string
    """
    return '{0}{{:{1}>{2}}}'.format(pre, spacer, length).format(bin(num)[2:])

def noisy_cutqc(circuit, data):
    circuit.remove_final_measurements()
    circuit_type = 'test_1q'
    circuit_size = circuit.num_qubits
    cutqc = CutQC(
        name="%s_%d" % (circuit_type, circuit_size),
        circuit=circuit,
        cutter_constraints={
            "max_subcircuit_width": circuit.num_qubits,
            "max_subcircuit_cuts": circuit.num_qubits*5,
            "subcircuit_size_imbalance": circuit.num_qubits,
            "max_cuts": circuit.num_qubits*5,
            "num_subcircuits": [2],
        },
        verbose=True,
    )
    cutqc.cut()
    for crc in cutqc.subcircuits:
        data.append(get_esp(crc, noise_model, backend))
    cutqc.evaluate(eval_mode="qasm_noisy", num_shots_fn=None, noise_model_fn = noise_model, backend_config_fn = backend)
    cutqc.build(mem_limit=1024, recursion_depth=100)
    st, error = cutqc.verify()
    s = sum(st)
    st = st/s
    counts = {}
    for i in range(0, len(st)):
        bin1 = binary(i, '', circuit.num_qubits, 0)
        counts[bin1] = st[i]*32768
    cutqc.clean_data()
    return collections.Counter(counts)

def noisy_cutqc_improved(circuit, data):
    circuit.remove_final_measurements()
    circuit_type = 'test_1q1'
    circuit_size = circuit.num_qubits
    cutqc = CutQC(
        name="%s_%d" % (circuit_type, circuit_size),
        circuit=circuit,
        cutter_constraints={
            "max_subcircuit_width": circuit.num_qubits,
            "max_subcircuit_cuts": circuit.num_qubits*5,
            "subcircuit_size_imbalance": circuit.num_qubits,
            "max_cuts": circuit.num_qubits*5,
            "num_subcircuits": [2],
        },
        verbose=True,
    )
    cutqc.cut()
    for crc in cutqc.subcircuits:
        data.append(get_esp(crc, noise_model, backend))
    cutqc.evaluate(eval_mode="qasm_noisy_mitigated", num_shots_fn=None, noise_model_fn = noise_model, backend_config_fn = backend)
    cutqc.build(mem_limit=1024, recursion_depth=100)
    st, error = cutqc.verify()
    s = sum(st)
    st = st/s
    counts = {}
    for i in range(0, len(st)):
        bin1 = binary(i, '', circuit.num_qubits, 0)
        counts[bin1] = st[i]*32768
    cutqc.clean_data()
    return collections.Counter(counts)

def ideal_simulation(circuit, shots=32768):
    """Simulate a circuit without noise (ideal case).
    
    Args:
        circuit: QuantumCircuit to simulate
        shots: Number of shots (default: 32768)
        
    Returns:
        Counter of measurement results
    """
    circuit.remove_final_measurements()
    circuit.measure_all()
    result = execute(
        circuit,
        Aer.get_backend('qasm_simulator'),
        shots=shots,
        simulator_seed=10
    ).result()
    counts1 = result.get_counts(circuit)
    circuit.remove_final_measurements()
    return collections.Counter(counts1)

def renormalize(ip_vector):
    """Renormalize a probability vector to sum to 1.
    
    Args:
        ip_vector: Dictionary of probabilities
        
    Returns:
        Renormalized dictionary
    """
    total = sum(ip_vector.values())
    if total > 0:
        for key in ip_vector.keys():
            ip_vector[key] = ip_vector[key] / total
    return ip_vector

def extrapolate_each_vec(ip_vector, esp, profiled_ip):
    op_dict = {}
    for key in ip_vector.keys():
        data = np.array([[1-esp, ip_vector[key]], [1, profiled_ip[key]]])
        fit = np.polyfit(data[:,0], data[:,1] ,1)
        line = np.poly1d(fit)
        op_dict[key] = (line([0])[0])
        if (op_dict[key] < 0):
            op_dict[key] = 0
    op_dict = renormalize(op_dict)
    return collections.Counter(op_dict)

def get_profiled_state(nqbs, tt):
    vqe = supermarq.hamiltonian_simulation.HamiltonianSimulation(nqbs, 1, tt)
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

def fold_gates(circuit, layers):
    circ = circuit.copy()
    circ.remove_final_measurements()
    for i in range (0, layers):
        for k in range(1, circ.num_qubits):
            circ.cx(k-1, k)
            circ.cx(k-1, k)
    return circ

def exponential2(x, k, b, m):
    return k*np.exp(-m*x/t1_time)+b

def exponential(x, k, b):
    return 1-k*np.exp(-x/t1_time)+b


def extrap_everything_esp(ip_vector, profile_ip, esp):
    """Extrapolate using ESP (Expected Success Probability).
    
    Args:
        ip_vector: Input vector to be mitigated
        profile_ip: Profiled vector
        esp: ESP of the circuit
        
    Returns:
        Mitigated statevector as Counter
    """
    op_dict = {}
    for key in ip_vector.keys():
        popt_exponential, pcov_exponential = scipy.optimize.curve_fit(
            exponential, [1, 1-esp], [profile_ip[key], ip_vector[key]], p0=[1, -0.5]
        )
        op_dict[key] = exponential(0, popt_exponential[0], popt_exponential[1])
    op_dict = renormalize(op_dict)
    return collections.Counter(op_dict)

def extrap_everything_time(ip_vector, profile_ip, t2, t1):
    op_dict = {}
    for key in ip_vector.keys():
        popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, [t2, t1], [profile_ip[key], ip_vector[key]], p0=[-1, 1], maxfev = 10000)
        op_dict[key] = exponential(0, popt_exponential[0], popt_exponential[1])
        if (op_dict[key] < 0):
            op_dict[key] = 0
    op_dict = renormalize(op_dict)
    return collections.Counter(op_dict)


def get_time(circuit, backend, lvl=0):
    """Calculate the maximum execution time for a circuit.
    
    Args:
        circuit: QuantumCircuit to analyze
        backend: Backend configuration
        lvl: Optimization level (default: 0)
        
    Returns:
        Maximum execution time in microseconds
    """
    circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=lvl,
        scheduling_method='asap'
    )
    max_time = 0
    us = 1e6
    config = backend.configuration()
    for i in range(16):
        t = circuit.qubit_duration(i) * config.dt * us
        if t > max_time:
            max_time = t
    return max_time

def get_profiled_state_decoh(nqbs, tt, backend, true_prof = True):
    if true_prof:
        vqe = supermarq.hamiltonian_simulation.HamiltonianSimulation(nqbs, 1, tt)
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
        profiled_ip = renormalize(profiled_ip)
        time = get_time(circuit, backend, 0)
    else:
        time = 20*t1_time
        getbin = lambda x, n: format(x, 'b').zfill(n)
        profiled_ip = {}
        for i in range(0, 2**nqbs):
            if i != 0:
                profiled_ip[getbin(i, nqbs)] = 1/(2**nqbs)
            else:
                profiled_ip[getbin(i, nqbs)] = 1 + 1/(2**nqbs)
        profiled_ip = collections.Counter(profiled_ip)
        profiled_ip = renormalize(profiled_ip)
    return profiled_ip, time

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
            zz = 1
        else: 
            profile_ip[key] = 0
        getbin = lambda x, n: format(x, 'b').zfill(n)  
        try:
            popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential2, t, c, p0=[-1.1, 1.2, 1.1], maxfev = 10000)
            op_dict[key] = exponential2(0, popt_exponential[0], popt_exponential[1], popt_exponential[2])
        except (RuntimeError):
            op_dict[key] = c[1]
        if (op_dict[key] < 0):
            op_dict[key] = 0
    print(op_dict)
    op_dict = renormalize(op_dict)
    return collections.Counter(op_dict)

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

def value_getter(item):
    return item[1]

def extrap_everything_depol(ip_vector, esp, profiled_ip):
    op_dict = {}
    for key in ip_vector.keys():
        data = np.array([[1-esp, ip_vector[key]], [1, profiled_ip[key]]])
        fit = np.polyfit(data[:,0], data[:,1] ,1)
        line = np.poly1d(fit)
        op_dict[key] = (line([0])[0])
        if (op_dict[key] < 0):
            op_dict[key] = 0
    op_dict = renormalize(op_dict)
    return collections.Counter(op_dict)

def extrap_everything_depol_N(ip_vector, esp, profiled_ip, N):
    op_dict = {}
    sval = sorted(ip_vector.items(), key = value_getter, reverse = True)
    for i in range(0, N):
        key = sval[i][0]
        data = np.array([[1-esp, ip_vector[key]], [1, profiled_ip[key]]])
        fit = np.polyfit(data[:,0], data[:,1] ,1)
        line = np.poly1d(fit)
        op_dict[key] = (line([0])[0])
        if (op_dict[key] < 0):
            op_dict[key] = 0
    
    for i in range(N, len(sval)):
        key = sval[i][0]
        op_dict[key] = sval[i][1]
    op_dict = renormalize(op_dict)
    return collections.Counter(op_dict)

def extrap_everything_depol_N_r(ip_vector, esp, profiled_ip, N):
    op_dict = {}
    sval = sorted(ip_vector.items(), key = value_getter, reverse = True)
    for i in range(0, N):
        key = sval[i][0]
        data = np.array([[1-esp, ip_vector[key]], [1, profiled_ip[key]]])
        fit = np.polyfit(data[:,0], data[:,1] ,1)
        line = np.poly1d(fit)
        op_dict[key] = (line([0])[0])
        if (op_dict[key] < 0):
            op_dict[key] = 0
    
    for i in range(N, len(sval)):
        key = sval[i][0]
        op_dict[key] = sval[i][1]
    op_dict = renormalize_top(op_dict, N)
    return collections.Counter(op_dict)

def extrap_everything_depol_N_r2(ip_vector, esp, profiled_ip, N):
    op_dict = {}
    sval = sorted(ip_vector.items(), key = value_getter, reverse = True)
    for i in range(0, N):
        key = sval[i][0]
        data = np.array([[1-esp, ip_vector[key]], [1, profiled_ip[key]]])
        fit = np.polyfit(data[:,0], data[:,1] ,1)
        line = np.poly1d(fit)
        op_dict[key] = (line([0])[0])
        if (op_dict[key] < 0):
            op_dict[key] = 0
    for i in range(N, len(sval)):
        key = sval[i][0]
        op_dict[key] = sval[i][1]
        
    op_dict = renorm_ratio(op_dict, ip_vector, N)
    return collections.Counter(op_dict)

def renormalize_top(op_dict, N):
    top_add = 0
    bot_add = 0
    sval = sorted(op_dict.items(), key = value_getter, reverse = True)
    for i in range(0, N):
        key = sval[i][0]
        top_add += op_dict[key]
    
    for i in range(N, len(sval)):
        key = sval[i][0]
        bot_add += op_dict[key]
    
    mul_fac = bot_add/top_add
    for i in range(N, len(sval)):
        key = sval[i][0]
        op_dict[key] = mul_fac * op_dict[key]
    return renormalize(op_dict)

def renorm_ratio(op_dict, in_dict, N):
    top_add = 0
    sval = sorted(op_dict.items(), key = value_getter, reverse = True)
    for i in range(0, N):
        key = sval[i][0]
        top_add += op_dict[key]/in_dict[key]
    mul_fac = N/top_add
    for i in range(N, len(sval)):
        key = sval[i][0]
        op_dict[key] = mul_fac * op_dict[key]
    return renormalize(op_dict)

def get_arithmatic_mean(depol_vec, decoh_vec):
    ret_vec = {}
    for key in depol_vec.keys():
        ret_vec[key] = (depol_vec[key] + decoh_vec[key])/2
    ret_vec = renormalize(ret_vec)
    return ret_vec

def geometric_mean(depol_vec, decoh_vec):
    ret_vec = {}
    for key in depol_vec.keys():
        ret_vec[key] = np.sqrt(depol_vec[key] * decoh_vec[key])
    ret_vec = renormalize(ret_vec)
    return ret_vec

def get_DZNE_extrapolated_state(circuit):
    folded2 = fold_gates_at_random(circuit, scale_factor=2)
    folded3 = fold_gates_at_random(circuit, scale_factor=3)
    folded4 = fold_gates_at_random(circuit, scale_factor=4)
    folded5 = fold_gates_at_random(circuit, scale_factor=5)
    fold1 = noise_simulation(circuit)
    fold2 = noise_simulation(folded2)
    fold3 = noise_simulation(folded3)
    fold4 = noise_simulation(folded4)
    fold5 = noise_simulation(folded5)
    NOISE_LEVELS = [1.0, 2.0, 3.0, 4.0, 5.0]
    fac = LinearFactory(NOISE_LEVELS)
    extr = {}
    for key in fold1.keys():
        if key in fold2.keys():
            if key in fold3.keys():
                extr[key] = fac.extrapolate(scale_factors = NOISE_LEVELS, exp_values = [fold1[key], fold2[key], fold3[key], fold4[key], fold5[key]])
                if extr[key] < 0:
                    extr[key] = 0
            else:
                extr[key] = fold1[key]
        else:
            extr[key] = fold1[key]
    extr = renormalize(extr)
    
    return collections.Counter(extr)



#The function process_data() is from the code of QRAFT
def process_data(RUN):
	output=""
	# Get the day and run ID
	day = RUN.split('/')[2]
	run = RUN.split('_')[2].split('.')[0]

	# Process the metadata
	with open('../data_experimental/' + day + '/computername_' + run + '.json', 'r') as f:
		cn = machine_list.index(json.load(f))

	with open('../data_experimental/' + day + '/circuitwidth_' + run + '.json', 'r') as f:
		wd = json.load(f)

	with open('../data_experimental/' + day + '/circuitdepth_' + run + '.json', 'r') as f:
		dp = json.load(f)

	with open('../data_experimental/' + day + '/countopgates_' + run + '.json', 'r') as f:
		co = json.load(f)
		for gate in ['x', 'sx', 'rz', 'cx']:
			if gate not in co:
				co[gate] = 0

	with open('../data_experimental/' + day + '/outstateprob_' + run + '.json', 'r') as f:
		sp = json.load(f)
		hw = {k:k.count('1') for k in sp.keys()}

	# Get the up and up + dn runs
	runs_up = glob.glob('../data_experimental/' + day + '/upcircprob_*_' + run + '.json')
	runs_up_dn = glob.glob('../data_experimental/' + day + '/updncrprob_*_' + run + '.json')


	# Process up runs data
	data_up = {k:[] for k in sp.keys()}
	for run in runs_up:
		with open(run, 'r') as f:
			dt = json.load(f)

		for k, v in data_up.items():
			if k in dt:
				v.append(dt[k])
			else:
				v.append(0)

	p25_up = {k:np.percentile(v, 25) for k, v in data_up.items()}
	p50_up = {k:np.percentile(v, 50) for k, v in data_up.items()}
	p75_up = {k:np.percentile(v, 75) for k, v in data_up.items()}

	e25_up = {k:np.percentile([p-sp[k] for p in v], 25) for k, v in data_up.items()}
	e50_up = {k:np.percentile([p-sp[k] for p in v], 50) for k, v in data_up.items()}
	e75_up = {k:np.percentile([p-sp[k] for p in v], 75) for k, v in data_up.items()}

	# Process up + dn runs data
	data_up_dn = {k:[] for k in sp.keys()}
	for run in runs_up_dn:
		with open(run, 'r') as f:
			dt = json.load(f)

		for k, v in data_up_dn.items():
			if k in dt:
				v.append(dt[k])
			else:
				v.append(0)

	e25_up_dn = {k:np.percentile(v, 25) for k, v in data_up_dn.items()}
	e50_up_dn = {k:np.percentile(v, 50) for k, v in data_up_dn.items()}
	e75_up_dn = {k:np.percentile(v, 75) for k, v in data_up_dn.items()}

	e25_up_dn[list(sp.keys())[0]] = np.percentile([p - 1 for p in data_up_dn[list(sp.keys())[0]]], 25)
	e50_up_dn[list(sp.keys())[0]] = np.percentile([p - 1 for p in data_up_dn[list(sp.keys())[0]]], 50)
	e75_up_dn[list(sp.keys())[0]] = np.percentile([p - 1 for p in data_up_dn[list(sp.keys())[0]]], 75)

	t25_up_dn = np.percentile([1 - p for p in data_up_dn[list(sp.keys())[0]]], 25)
	t50_up_dn = np.percentile([1 - p for p in data_up_dn[list(sp.keys())[0]]], 50)
	t75_up_dn = np.percentile([1 - p for p in data_up_dn[list(sp.keys())[0]]], 75)

	# Generate the general header
	header =  str(cn) + ','
	header += str(wd) + ','
	header += str(dp) + ','
	header += str(co['x']) + ','
	header += str(co['sx']) + ','
	header += str(co['rz']) + ','
	header += str(co['cx']) + ','
	header += str(int(t25_up_dn*100)) + ','
	header += str(int(t50_up_dn*100)) + ','
	header += str(int(t75_up_dn*100)) + ','

	# Form the strings for each state
	for k in sp.keys():

		# Make the general string for each state
		string =  header + str(hw[k]) + ','
		string += str(int(p25_up[k]*100)) + ','
		string += str(int(p50_up[k]*100)) + ','
		string += str(int(p75_up[k]*100)) + ','
		string += str(int(e25_up_dn[k]*100)) + ','
		string += str(int(e50_up_dn[k]*100)) + ','
		string += str(int(e75_up_dn[k]*100)) + ','

		# Make the strings for different error percentiles
		output += string + str(int(sp[k]/sum(sp.values())*100)) + '\n'  
        
	return output


def collect_qraft_data(qc: QuantumCircuit, current_test: str = current_test, shots: int = M):
    ff = str(len(glob.glob('../data_experimental/' + today + current_test + '/computername_*.json')))
   
    n=qc.num_qubits
    
    sp = execute(qc, Aer.get_backend('statevector_simulator')).result()
    sp = [abs(p)**2 for p in sp.get_statevector(qc)]
    sp = {S[n][s]:sp[s] for s in range(2**n)}
    
    
    qc_dn = qc.inverse()
    qc_up_dn = qc.combine(qc_dn)
    
    '''for i in range(n):
        qc.measure(i, i)
        qc_up_dn.measure(i, i)'''
    qc.measure_all()
    qc_up_dn.measure_all()
    
    qc = transpile(qc, 
                        basis_gates=basis_gates,
                        coupling_map=coupling_map,
                        optimization_level=0,
                        seed_transpiler=0,
                       )
    counts = execute(
            experiments=qc,
            backend=Aer.get_backend("qasm_simulator"),
            noise_model=noise_model,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            optimization_level=0,  
            seed_simulator=0,
            shots=shots,
        ).result().get_counts()
    
    
    counts_up_dn = execute(
            experiments=qc_up_dn,
            backend=Aer.get_backend("qasm_simulator"),
            noise_model=noise_model,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            optimization_level=0,  
            seed_simulator=0,
            shots=shots,
        ).result().get_counts()
        
    o_up = json.dumps({k:v/M for k, v in counts.items()})
    o_up_dn = json.dumps({k:v/shots for k, v in counts_up_dn.items()})
    
    cn = json.dumps(C[0])
    wd = json.dumps(n) 
    dp = json.dumps(qc.depth())
    co = json.dumps(dict(qc.count_ops()))
    sp = json.dumps(sp)
    
    with open('../data_experimental/' + today + current_test + '/computername_' + ff + '.json', 'w') as fl:
        fl.write(cn)
    with open('../data_experimental/' + today + current_test + '/circuitwidth_' + ff + '.json', 'w') as fl:
        fl.write(wd)
    with open('../data_experimental/' + today + current_test + '/circuitdepth_' + ff + '.json', 'w') as fl:
        fl.write(dp)
    with open('../data_experimental/' + today + current_test + '/countopgates_' + ff + '.json', 'w') as fl:
        fl.write(co)
    with open('../data_experimental/' + today + current_test + '/outstateprob_' + ff + '.json', 'w') as fl:
        fl.write(sp)
    with open('../data_experimental/' + today + current_test + '/upcircprob_0_' + ff + '.json', 'w') as fl:
        fl.write(o_up)
    with open('../data_experimental/' + today + current_test + '/updncrprob_0_' + ff + '.json', 'w') as fl:
        fl.write(o_up_dn)


def get_expectation_qraft(qc: QuantumCircuit, current_test: str = current_test, shots: int = M):
    if os.path.exists('../data_experimental/' + today + current_test):
        dir = '../data_experimental/' + today + current_test
        filelist = glob.glob(os.path.join(dir, "*"))

        for f in filelist:
            os.remove(f)
    else:
        os.makedirs('../data_experimental/' + today + current_test)
    collect_qraft_data(qc,current_test)
    
    RUN = glob.glob('../data_experimental/' + today + current_test + '/computername_0.json')[0]
    outputs=process_data(RUN)
    
    day = RUN.split('/')[2]
    run = RUN.split('_')[2].split('.')[0]
    
    nqubits = qc.num_qubits
    predict_result=[] #save the predict result
    for i in range(0,2**nqubits):
        out=outputs.split('\n')[i]
        data_list=[] #save the data needed in order to predict
        for j in range(0,17):
            data=float(out.split(',')[j])
            data_list.append(data)
        
        tblIn = matlab.double(data_list)
        ypredOut = eng.mypredict1(tblIn)
        predict_result.append(ypredOut)
    
    #do normalization for the predict_result
    sum=0
    normalization_results=[]
    for result in predict_result:
        sum+=result
    if(sum!=0):
        for result in predict_result:
            result=result/sum
            normalization_results.append(result)
    
    #represent normalization_results using dict
    counts_value={}
    if(len(normalization_results)>0):
        for i in range(2**nqubits):
            counts_value[S[nqubits][i]]=normalization_results[i]
    else:
        with open('../data_experimental/' + day + '/upcircprob_0_' + run + '.json', 'r') as f: 
            counts_value = json.load(f)
    
    #use the normalization_results
    #print(counts_value)
    
    return renormalize(counts_value)

rule = ddd.rules.xyxy

def get_DZNE_extrap(circuit, vqe):
    def ibmq_executor(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        
        circuit.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        #quasis = mit.apply_correction(counts, range(circuit.num_qubits))
        #counts1 = quasis.nearest_probability_distribution()
        return vqe._average_magnetization(collections.Counter(renormalize(counts)), 1)
    NOISE_LEVELS = [1.0, 2.0, 3.0, 4.0, 5.0]
    ef = LinearFactory(NOISE_LEVELS)
    mitigated = zne.execute_with_zne(circuit, ibmq_executor, factory = ef)
    return mitigated

def get_DZNE_full_state_extrap(circuit, vqe):
    extrap = get_DZNE_extrapolated_state(circuit)
    return vqe._average_magnetization(extrap, 1), extrap

def get_cdr(circuit, vqe):
    def ibmq_executor(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        #print(circuit)
        circuit.remove_final_measurements()
        circuit.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        #quasis = mit.apply_correction(counts, range(circuit.num_qubits))
        #counts1 = quasis.nearest_probability_distribution()
        #print(circuit)
        return vqe._average_magnetization(collections.Counter(renormalize(counts)), 1)
    def ibmq_executor_nless(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'), shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        #quasis = mit.apply_correction(counts, range(circuit.num_qubits))
        #counts1 = quasis.nearest_probability_distribution()
        #print(circuit)
        return vqe._average_magnetization(collections.Counter(renormalize(counts)), 1)
    mitigated_measurement = cdr.execute_with_cdr(circuit, ibmq_executor, observable=None, simulator=ibmq_executor_nless, seed=0, num_training_circuits=64).real
    return mitigated_measurement

def get_vncdr(circuit, vqe):
    def ibmq_executor(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        #quasis = mit.apply_correction(counts, range(circuit.num_qubits))
        #counts1 = quasis.nearest_probability_distribution()
        #print(circuit)
        return vqe._average_magnetization(collections.Counter(renormalize(counts)), 1)
    def ibmq_executor_nless(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'), shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        #quasis = mit.apply_correction(counts, range(circuit.num_qubits))
        #counts1 = quasis.nearest_probability_distribution()
        #print(circuit)
        return vqe._average_magnetization(collections.Counter(renormalize(counts)), 1)
    mitigated_measurement = cdr.execute_with_cdr(circuit, ibmq_executor, observable=None, 
                                                 simulator=ibmq_executor_nless, seed=0, num_training_circuits=64, 
                                                 scale_factors=(1.0, 2.0, 3.0, 4.0)).real
    return mitigated_measurement

def get_dd(circuit, vqe):
    def ibmq_executor2(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        #quasis = mit.apply_correction(counts, range(circuit.num_qubits))
        #counts1 = quasis.nearest_probability_distribution()
        #print(circuit)
        return vqe._average_magnetization(collections.Counter(counts), 1)
    mitigated_result = ddd.execute_with_ddd(circuit=circuit, executor=ibmq_executor2, rule=rule)
    return mitigated_result

def get_absolute_error(ideal, actual):
    """Calculate absolute error between ideal and actual values.
    
    Args:
        ideal: Ideal/expected value
        actual: Actual/measured value
        
    Returns:
        Absolute error
    """
    return abs(ideal - actual)

def extrap_everything_decoh_5_N(ip_vector, circuit, N):
    op_dict = {}
    sval = sorted(ip_vector.items(), key = value_getter, reverse = True)
    t = get_time(circuit, backend, 0)
    
    getbin = lambda x, n: format(x, 'b').zfill(n)
    for i in range(0, N):
        key = sval[i][0]
        ones = count_ones(key)
        if (key == getbin(0, circuit.num_qubits)):
            op_dict[key] = (1 - ip_vector[key])/(np.exp(-t/t1_time))
        else:
            op_dict[key] = ip_vector[key]/(np.exp(-ones*t/t1_time))
        
    op_dict = renormalize_top(op_dict, N)
    return collections.Counter(op_dict)

def count_ones(inp):
    return inp.count('1')

def extrap_everything_decoh_5(noise_sim, t_1, n):
    op_dict = {}
    circ1 = [noise_sim]
    t = [t_1]
    
    getbin = lambda x, n: format(x, 'b').zfill(n)
    for key in circ1[0].keys():
        ones = count_ones(key)
        #print(key, ones, circuit.num_qubits)
        if (key == getbin(0, n)):
            #print(key, circ1[0][key])
            op_dict[key] = (1 - circ1[0][key])/(np.exp(-t[0]/t1_time))
            #print(key, op_dict[key])
        else:
            op_dict[key] = circ1[0][key]/(np.exp(-ones*t[0]/t1_time))
            #print(key, ones, circ1[0][key], np.exp(-ones*t[0]/t1_time), (np.exp(-t[0]/t1_time)))
    op_dict = renormalize(op_dict)
    
    #print(circ1[0], collections.Counter(op_dict))
    return collections.Counter(op_dict)

def extrap_everything_decoh_4(noise_sim, t_1, n):
    op_dict = {}
    circ1 = [noise_sim]
    t = [t_1]
    
    getbin = lambda x, n: format(x, 'b').zfill(n)
    for key in circ1[0].keys():
        ones = count_ones(key)
        #print(key, ones, circuit.num_qubits)
        if (0):
            #print(key, circ1[0][key])
            op_dict[key] = (1 - circ1[0][key])/(np.exp(-t[0]/t1_time))
            #print(key, op_dict[key])
        else:
            op_dict[key] = circ1[0][key]/(np.exp(-ones*t[0]/t1_time))
            #print(key, ones, circ1[0][key], np.exp(-ones*t[0]/t1_time), (np.exp(-t[0]/t1_time)))
    op_dict = renormalize(op_dict)
    
    #print(circ1[0], collections.Counter(op_dict))
    return collections.Counter(op_dict)

def extrap_everything_decoh_3(noise_sim, t_1):
    op_dict = {}
    circ1 = [noise_sim]
    t = [t_1]
    
    getbin = lambda x, n: format(x, 'b').zfill(n)
    for key in circ1[0].keys():
        if (circ1[0][key] == 0):
            op_dict[key] = (1 - circ1[0][key])/(np.exp(-t[0]/t1_time))
        else:
            op_dict[key] = circ1[0][key]/(np.exp(-t[0]/t1_time))
    op_dict = renormalize(op_dict)
    
    #print(ip_vector, op_dict)
    return collections.Counter(op_dict)

def save_object(obj, filename):
    """Save an object to a pickle file.
    
    Args:
        obj: Object to save
        filename: Path to output file
    """
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Load an object from a pickle file.
    
    Args:
        filename: Path to input file
        
    Returns:
        Loaded object
    """
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

'''noise_model = load_object("noisemodel.pkl")
print(noise_model)'''
def get_DZNE_QAOA_extrap(circuit, vqe):
    def ibmq_executor(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        #quasis = mit.apply_correction(counts, range(circuit.num_qubits))
        #counts1 = quasis.nearest_probability_distribution()
        #print(circuit)
        return vqe._get_expectation_value_from_probs(renormalize(collections.Counter(counts)))
    NOISE_LEVELS = [1.0, 2.0, 3.0, 4.0, 5.0]
    ef = LinearFactory(NOISE_LEVELS)
    mitigated = zne.execute_with_zne(circuit, ibmq_executor, factory = ef)
    return mitigated

def get_DZNE_QAOA_fstextr(circuit, vqe):
    dzne = get_DZNE_extrapolated_state(circuit)
    return vqe._get_expectation_value_from_probs(dzne), dzne

def get_cdr_QAOA(circuit, vqe):
    def ibmq_executor(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        #quasis = mit.apply_correction(counts, range(circuit.num_qubits))
        #counts1 = quasis.nearest_probability_distribution()
        #print(circuit)
        return vqe._get_expectation_value_from_probs(renormalize(collections.Counter(counts)))
    def ibmq_executor_nless(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'), 
                             shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        #quasis = mit.apply_correction(counts, range(circuit.num_qubits))
        #counts1 = quasis.nearest_probability_distribution()
        #print(circuit)
        return vqe._get_expectation_value_from_probs(renormalize(collections.Counter(counts)))
    mitigated_measurement = cdr.execute_with_cdr(circuit, ibmq_executor, observable=None, simulator=ibmq_executor_nless, 
                                                 seed=0, num_tarining_circuits=64).real
    return mitigated_measurement

def get_vncdr_QAOA(circuit, vqe):
    def ibmq_executor(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        #quasis = mit.apply_correction(counts, range(circuit.num_qubits))
        #counts1 = quasis.nearest_probability_distribution()
        #print(circuit)
        return vqe._get_expectation_value_from_probs(renormalize(collections.Counter(counts)))
    def ibmq_executor_nless(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'), 
                             shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        #quasis = mit.apply_correction(counts, range(circuit.num_qubits))
        #counts1 = quasis.nearest_probability_distribution()
        #print(circuit)
        return vqe._get_expectation_value_from_probs(renormalize(collections.Counter(counts)))
    mitigated_measurement = cdr.execute_with_cdr(circuit, ibmq_executor, observable=None, simulator=ibmq_executor_nless, 
                                                 scale_factors=(1.0, 2.0, 3.0, 4.0), num_tarining_circuits=64,seed=0).real
    return mitigated_measurement


def get_ddd_QAOA(circuit, vqe):
    def ibmq_executor2(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        #quasis = mit.apply_correction(counts, range(circuit.num_qubits))
        #counts1 = quasis.nearest_probability_distribution()
        #print(circuit)
        return vqe._get_expectation_value_from_probs(renormalize(collections.Counter(counts)))
    mitigated_result = ddd.execute_with_ddd(circuit=circuit, executor=ibmq_executor2, rule=rule)
    return mitigated_result

def get_DZNE_VQE_extrap(circuitz, vqe):
    def ibmq_executor(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit2 = copy.deepcopy(circuit)
        for i in range(0, circuit.num_qubits):
            circuit2.h(i)
        circuit.measure_all()
        circuit2.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        counts = renormalize(collections.Counter(counts))

        job = qiskit.execute(circuit2, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts2 = job.result().get_counts()
        counts2 = renormalize(collections.Counter(counts2))

        return vqe._get_expectation_value_from_probs(counts, counts2)
    NOISE_LEVELS = [1.0, 2.0, 3.0, 4.0, 5.0]
    ef = LinearFactory(NOISE_LEVELS)
    mitigated = zne.execute_with_zne(circuitz, ibmq_executor, factory = ef)
    return mitigated

def get_DZNE_VQE_fstextr(circuit, circuitz, vqe):
    extrap = get_DZNE_extrapolated_state(circuit)
    extrapz = get_DZNE_extrapolated_state(circuitz)
    return vqe._get_expectation_value_from_probs(extrapz, extrap), extrap, extrapz

def get_cdr_VQE(circuitz, vqe):
    def ibmq_executor(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit2 = copy.deepcopy(circuit)
        for i in range(0, circuit.num_qubits):
            circuit2.h(i)
        circuit.measure_all()
        circuit2.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        counts = renormalize(collections.Counter(counts))

        job = qiskit.execute(circuit2, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts2 = job.result().get_counts()
        counts2 = renormalize(collections.Counter(counts2))
        return vqe._get_expectation_value_from_probs(counts, counts2)
    def ibmq_executor_nless(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit2 = copy.deepcopy(circuit)
        for i in range(0, circuit.num_qubits):
            circuit2.h(i)
        circuit.measure_all()
        circuit2.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
                             shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        counts = renormalize(collections.Counter(counts))

        job = qiskit.execute(circuit2, Aer.get_backend('qasm_simulator'),
                             shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts2 = job.result().get_counts()
        counts2 = renormalize(collections.Counter(counts2))

        return vqe._get_expectation_value_from_probs(counts, counts2)
    mitigated_measurement = cdr.execute_with_cdr(circuitz, ibmq_executor, observable=None, simulator=ibmq_executor_nless, 
                                                 seed=0, num_training_circuits=64).real
    return mitigated_measurement

def get_vncdr_VQE(circuitz, vqe):
    def ibmq_executor(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit2 = copy.deepcopy(circuit)
        for i in range(0, circuit.num_qubits):
            circuit2.h(i)
        circuit.measure_all()
        circuit2.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        counts = renormalize(collections.Counter(counts))

        job = qiskit.execute(circuit2, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts2 = job.result().get_counts()
        counts2 = renormalize(collections.Counter(counts2))
        return vqe._get_expectation_value_from_probs(counts, counts2)
    def ibmq_executor_nless(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit2 = copy.deepcopy(circuit)
        for i in range(0, circuit.num_qubits):
            circuit2.h(i)
        circuit.measure_all()
        circuit2.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
                             shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        counts = renormalize(collections.Counter(counts))

        job = qiskit.execute(circuit2, Aer.get_backend('qasm_simulator'),
                             shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts2 = job.result().get_counts()
        counts2 = renormalize(collections.Counter(counts2))

        return vqe._get_expectation_value_from_probs(counts, counts2)
    mitigated_measurement = cdr.execute_with_cdr(circuitz, ibmq_executor, observable=None, simulator=ibmq_executor_nless, 
                                                 seed=0, num_training_circuits=64, scale_factors=(1.0, 2.0, 3.0, 4.0, 5.0)).real
    return mitigated_measurement

def get_ddd_VQE(circuitz, vqe):
    def ibmq_executor2(circuit: qiskit.QuantumCircuit, shots: int = 32768) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
            shots: Number of times to execute the circuit to compute the expectation value.
        """
        circuit.remove_final_measurements()
        circuit2 = copy.deepcopy(circuit)
        for i in range(0, circuit.num_qubits):
            circuit2.h(i)
        circuit.measure_all()
        circuit2.measure_all()
        job = qiskit.execute(circuit, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts = job.result().get_counts()
        counts = renormalize(collections.Counter(counts))

        job = qiskit.execute(circuit2, Aer.get_backend('qasm_simulator'),
             coupling_map=coupling_map,
             basis_gates=basis_gates,
             noise_model=noise_model, shots = shots, simulator_seed = 10, optimization_level=0)

        # Convert from raw measurement counts to the expectation value
        counts2 = job.result().get_counts()
        counts2 = renormalize(collections.Counter(counts2))

        return vqe._get_expectation_value_from_probs(counts, counts2)
    mitigated_result = ddd.execute_with_ddd(circuit=circuitz, executor=ibmq_executor2, rule=rule)
    return mitigated_result

def define_complete_vec(ip_vector, nqbs):
    getbin = lambda x, n: format(x, 'b').zfill(n)
    out_vec = {}
    for i in range(0, 2**nqbs):
        key = getbin(i, nqbs)
        if (key in ip_vector):
            out_vec[key] = ip_vector[key]
        else:
            out_vec[key] = 0.0
    return collections.Counter(out_vec)

def get_fidelity(idl_vec, comp_vec, nqbs):
    idl_vec = define_complete_vec(idl_vec, nqbs)
    comp_vec = define_complete_vec(comp_vec, nqbs)
    return hellinger_fidelity(idl_vec, comp_vec)

def execute_hamsimul_test(nqbs, tt, dirname):
    data = []
    print("Creating circuit qubits {} tt {} ...".format(str(nqbs), str(tt)))
    vqe = supermarq.hamiltonian_simulation.HamiltonianSimulation(nqbs, 1, tt)
    vqe_circuit = vqe.circuit()
    circuit = supermarq.converters.cirq_to_qiskit(vqe_circuit)
    circuit.remove_final_measurements()
    data.append(nqbs)
    data.append(tt)
    data.append(circuit.depth())
    filename = dirname + "/Hamsim_" + str(nqbs) + "_" + str(tt) + ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_object(vqe, filename)
    tim = get_time(circuit, backend, 0)
    profiled_ip2_depol= get_profiled_state_depol(nqbs, tt, backend, False)

    noise_sim = noise_simulation(circuit)
    idl = renormalize(ideal_simulation(circuit))
    dt = vqe._average_magnetization(noise_sim, 1)
    ideal = vqe._average_magnetization(idl, 1)
    esp = get_esp(circuit, noise_model, backend)
    if (esp < 0.10):
        return data
    apr1 = extrap_everything_decoh_5(noise_sim, tim, nqbs)
    apr1_vec = extrap_everything_depol(apr1, esp, profiled_ip2_depol)
    apr1 = vqe._average_magnetization(apr1_vec, 1)

    apr2 = extrap_everything_decoh_5_N(noise_sim, copy.deepcopy(circuit), 10)
    apr2_vec = extrap_everything_depol_N_r2(apr2, esp, profiled_ip2_depol, 10)
    apr2 = vqe._average_magnetization(apr2_vec, 1)

    apr3_vec = extrap_everything_depol_N_r2(noise_sim, esp, profiled_ip2_depol, 10)
    apr3 = vqe._average_magnetization(apr3_vec, 1)
    
    '''apr4 = extrap_everything_decoh_4(noise_sim, tim, nqbs)
    apr4_vec = extrap_everything_depol(apr4, esp, profiled_ip2_depol)
    apr4 = vqe._average_magnetization(apr4_vec, 1)
    
    apr5 = extrap_everything_decoh_3(noise_sim, tim, nqbs)
    apr5_vec = extrap_everything_depol(apr5, esp, profiled_ip2_depol)
    apr5 = vqe._average_magnetization(apr5_vec, 1)'''


    dzne, dzne_full_st = get_DZNE_full_state_extrap(copy.deepcopy(circuit), vqe)
    dzne_extrapol = get_DZNE_extrap(copy.deepcopy(circuit), vqe)
    cdr_data = get_cdr(copy.deepcopy(circuit), vqe)
    vncdr_data = get_vncdr(copy.deepcopy(circuit), vqe)
    
    ext4 = extrap_everything_depol(noise_sim, esp, profiled_ip2_depol)
    extp4 = vqe._average_magnetization(ext4, 1)
    
    qft = get_expectation_qraft(copy.deepcopy(circuit))
    qft_data = vqe._average_magnetization(qft, 1)
    
    data.append(esp)
    data.append(tim)
    data.append(dt)
    data.append(ideal)
    data.append(extp4)
    data.append(apr3)
    data.append(apr1)
    data.append(apr2)
    data.append(dzne)
    data.append(dzne_extrapol)
    data.append(cdr_data)
    data.append(vncdr_data)
    data.append(qft_data)

    data.append(get_absolute_error(ideal, dt))
    data.append(get_absolute_error(ideal, extp4))
    data.append(get_absolute_error(ideal, apr3))
    data.append(get_absolute_error(ideal, apr1))
    data.append(get_absolute_error(ideal, apr2))
    data.append(get_absolute_error(ideal, dzne))
    data.append(get_absolute_error(ideal, dzne_extrapol))
    data.append(get_absolute_error(ideal, cdr_data))
    data.append(get_absolute_error(ideal, vncdr_data))
    data.append(get_absolute_error(ideal, qft_data))

    data.append(get_fidelity(idl, noise_sim, nqbs))
    data.append(get_fidelity(idl, ext4, nqbs))
    data.append(get_fidelity(idl, apr3_vec, nqbs))
    data.append(get_fidelity(idl, apr1_vec, nqbs))
    data.append(get_fidelity(idl, apr2_vec, nqbs))
    data.append(get_fidelity(idl, dzne_full_st, nqbs))
    data.append(get_fidelity(idl, qft, nqbs))
 
    return data  

def execute_VQE(nqbs, tt, dirname):
    data = []
    vqe = supermarq.vqe_proxy.VQEProxy(nqbs, tt)
    vqe_circuit = vqe.circuit()
    circuitz = supermarq.converters.cirq_to_qiskit(vqe_circuit[0])
    circuit = supermarq.converters.cirq_to_qiskit(vqe_circuit[1])
    circuit.remove_final_measurements()
    circuitz.remove_final_measurements()
    data.append(nqbs) 
    data.append(tt) 
    data.append(circuit.depth()) 

    print("Creating circuit qubits {} tt {} ...".format(str(nqbs), str(tt)))
    filename = dirname + "/VQE_" + str(nqbs) + "_" + str(tt) + ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_object(vqe, filename)
    profiled_ip2_depol= get_profiled_state_depol(nqbs, tt, backend, False)
    
    noise_sim = noise_simulation(circuit)
    noise_simz = noise_simulation(circuitz)
    dt = vqe._get_expectation_value_from_probs(noise_simz, noise_sim)
    esp = get_esp(circuit, noise_model, backend)
    esp_z = get_esp(circuitz, noise_model, backend)
    if (esp < 0.10):
        return data

    ideal = vqe._get_expectation_value_from_probs(renormalize(ideal_simulation(circuitz)), renormalize(ideal_simulation(circuit)))
    tx = get_time(circuit, backend, 0)
    tz = get_time(circuitz, backend, 0)

    apr1 = extrap_everything_decoh_5(noise_sim, tx, nqbs)
    apr1_z = extrap_everything_decoh_5(noise_simz, tz, nqbs)
    apr1_vec = extrap_everything_depol(apr1, esp, profiled_ip2_depol)
    apr1_z = extrap_everything_depol(apr1_z, esp_z, profiled_ip2_depol)
    
    apr2 = extrap_everything_decoh_5_N(noise_sim, copy.deepcopy(circuit), 10)
    apr2_z = extrap_everything_decoh_5_N(noise_simz, copy.deepcopy(circuitz), 10)
    apr2_vec = extrap_everything_depol_N_r2(apr2, esp, profiled_ip2_depol, 10)
    apr2_z = extrap_everything_depol_N_r2(apr2_z, esp_z, profiled_ip2_depol, 10)

    '''apr3 = extrap_everything_decoh_4(noise_sim, tx, nqbs)
    apr3_z = extrap_everything_decoh_4(noise_simz, tz, nqbs)
    apr3_vec = extrap_everything_depol(apr3, esp, profiled_ip2_depol)
    apr3_z = extrap_everything_depol(apr3_z, esp_z, profiled_ip2_depol)
    
    apr6 = extrap_everything_decoh_3(noise_sim, tx, nqbs)
    apr6_z = extrap_everything_decoh_3(noise_simz, tz, nqbs)
    apr6_vec = extrap_everything_depol(apr6, esp, profiled_ip2_depol)
    apr6_z = extrap_everything_depol(apr6_z, esp_z, profiled_ip2_depol)'''

    
    apr4_vec = extrap_everything_depol(noise_sim, esp, profiled_ip2_depol)
    apr4_z = extrap_everything_depol(noise_simz, esp_z, profiled_ip2_depol)
    
    apr5_vec = extrap_everything_depol_N_r2(noise_sim, esp, profiled_ip2_depol, 10)
    apr5_z = extrap_everything_depol_N_r2(noise_simz, esp_z, profiled_ip2_depol, 10)
    
    apr1 = vqe._get_expectation_value_from_probs(apr1_z, apr1_vec)
    apr2 = vqe._get_expectation_value_from_probs(apr2_z, apr2_vec)
    apr4 = vqe._get_expectation_value_from_probs(apr4_z, apr4_vec)
    apr5 = vqe._get_expectation_value_from_probs(apr5_z, apr5_vec)
    dzne_fstex, dzne, dznez = get_DZNE_VQE_fstextr(copy.deepcopy(circuit), copy.deepcopy(circuitz), vqe)
    dzne_extrp = get_DZNE_VQE_extrap(copy.deepcopy(circuitz), vqe)
    cdr_data = get_cdr_VQE(copy.deepcopy(circuitz), vqe)
    vncdr_data = get_vncdr_VQE(copy.deepcopy(circuitz), vqe)
    qft = get_expectation_qraft(copy.deepcopy(circuit))
    qftz = get_expectation_qraft(copy.deepcopy(circuitz))
    qft_data = vqe._get_expectation_value_from_probs(qftz, qft)


    data.append(esp)
    data.append(esp_z)
    data.append(get_time(circuit, backend, 0))
    data.append(dt)
    data.append(ideal)
    data.append(apr4)
    data.append(apr5)
    data.append(apr1)
    data.append(apr2)
    data.append(dzne_fstex)
    data.append(dzne_extrp)
    data.append(cdr_data)
    data.append(vncdr_data)
    data.append(qft_data)

    data.append(get_absolute_error(ideal, dt))
    data.append(get_absolute_error(ideal, apr4))    
    data.append(get_absolute_error(ideal, apr5)) 
    data.append(get_absolute_error(ideal, apr1))
    data.append(get_absolute_error(ideal, apr2))  
    data.append(get_absolute_error(ideal, dzne_fstex)) 
    data.append(get_absolute_error(ideal, dzne_extrp)) 
    data.append(get_absolute_error(ideal, cdr_data)) 
    data.append(get_absolute_error(ideal, vncdr_data))
    data.append(get_absolute_error(ideal, qft_data))

    idl = renormalize(ideal_simulation(circuit))
    data.append(get_fidelity(idl, noise_sim, nqbs))
    data.append(get_fidelity(idl, apr4_vec, nqbs))
    data.append(get_fidelity(idl, apr5_vec, nqbs))
    data.append(get_fidelity(idl, apr1_vec, nqbs))
    data.append(get_fidelity(idl, apr2_vec, nqbs))
    data.append(get_fidelity(idl, dzne, nqbs))
    data.append(get_fidelity(idl, qft, nqbs))

    idl = renormalize(ideal_simulation(circuitz))
    data.append(get_fidelity(idl, noise_simz, nqbs))
    data.append(get_fidelity(idl, apr4_z, nqbs))
    data.append(get_fidelity(idl, apr5_z, nqbs))
    data.append(get_fidelity(idl, apr1_z, nqbs))
    data.append(get_fidelity(idl, apr2_z, nqbs))
    data.append(get_fidelity(idl, dznez, nqbs))
    data.append(get_fidelity(idl, qftz, nqbs))
    return data

def execute_QAOA(nqbs, rvar, dirname):
    data = []
    tt = 4
    
    vqe = supermarq.qaoa_vanilla_proxy.QAOAVanillaProxy(nqbs)
    vqe_circuit = vqe.circuit()
    circuit = supermarq.converters.cirq_to_qiskit(vqe_circuit)
    circuit.remove_final_measurements()
    filename = dirname + "/QAOA_" + str(nqbs) + "_" + str(rvar) + ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_object(vqe, filename)
    data.append(nqbs) 
    data.append(circuit.depth())
    
    print(filename)
    tim = get_time(circuit, backend, 0)
    profiled_ip2_depol = get_profiled_state_depol(nqbs, tt, backend, False)

    noise_sim = noise_simulation(circuit)
    dt = vqe._get_expectation_value_from_probs(noise_sim)
    esp = get_esp(circuit, noise_model, backend)
    if (esp < 0.10):
        return data
    idl = renormalize(ideal_simulation(circuit))
    ideal = vqe._get_expectation_value_from_probs(idl)
    
    extrapd2 = extrap_everything_depol(noise_sim, esp, profiled_ip2_depol)
    apr1 = extrap_everything_decoh_5(noise_sim, tim, nqbs)
    apr1_vec = extrap_everything_depol(apr1, esp, profiled_ip2_depol)
    apr1 = vqe._get_expectation_value_from_probs(apr1_vec)

    apr2 = extrap_everything_decoh_5_N(noise_sim, copy.deepcopy(circuit), 10)
    apr2_vec = extrap_everything_depol_N_r2(apr2, esp, profiled_ip2_depol, 10)
    apr2 = vqe._get_expectation_value_from_probs(apr2_vec)
    
    apr3_vec = extrap_everything_depol_N_r2(noise_sim, esp, profiled_ip2_depol, 10)
    apr3 = vqe._get_expectation_value_from_probs(apr3_vec)

    rzne_danly = vqe._get_expectation_value_from_probs(extrapd2)
    circuit.remove_final_measurements()
    dzne_extrp = get_DZNE_QAOA_extrap(copy.deepcopy(circuit), vqe)
    dzne_fstex, dzne_st = get_DZNE_QAOA_fstextr(copy.deepcopy(circuit), vqe)
    cdr_data = get_cdr_QAOA(copy.deepcopy(circuit), vqe)
    vncdr_data = get_vncdr_QAOA(copy.deepcopy(circuit), vqe)
    circuit.remove_final_measurements()
    
    qft = get_expectation_qraft(copy.deepcopy(circuit))
    qft_data = vqe._get_expectation_value_from_probs(qft)

    #data = data + str(circuit.count_ops()) 
    data.append(get_esp(circuit, noise_model, backend)) 
    data.append(tim)
    data.append(dt) 
    data.append(ideal) 
    data.append(rzne_danly)   
    data.append(apr3) 
    data.append(apr1)
    data.append(apr2) 
    data.append(dzne_extrp)
    data.append(dzne_fstex)
    data.append(cdr_data)
    data.append(vncdr_data)
    data.append(qft_data)
    data.append(get_absolute_error(ideal, dt))
    data.append(get_absolute_error(ideal, rzne_danly))
    data.append(get_absolute_error(ideal, apr3))
    data.append(get_absolute_error(ideal, apr1))
    data.append(get_absolute_error(ideal, apr2))
    data.append(get_absolute_error(ideal, dzne_extrp))
    data.append(get_absolute_error(ideal, dzne_fstex))
    data.append(get_absolute_error(ideal, cdr_data))
    data.append(get_absolute_error(ideal, vncdr_data))
    data.append(get_absolute_error(ideal, qft_data))
    data.append(get_fidelity(idl, noise_sim, nqbs))
    data.append(get_fidelity(idl, apr3_vec, nqbs))
    data.append(get_fidelity(idl, apr1_vec, nqbs))
    data.append(get_fidelity(idl, apr2_vec, nqbs))
    data.append(get_fidelity(idl, extrapd2, nqbs))
    data.append(get_fidelity(idl, dzne_st, nqbs))
    data.append(get_fidelity(idl, qft, nqbs))


    return data

def execute_QAOA_Swap(nqbs, rvar, dirname):
    data = []
    tt = 4
    
    vqe = supermarq.qaoa_fermionic_swap_proxy.QAOAFermionicSwapProxy(nqbs)
    vqe_circuit = vqe.circuit()
    circuit = supermarq.converters.cirq_to_qiskit(vqe_circuit)
    circuit.remove_final_measurements()
    filename = dirname + "/QAOASwap_" + str(nqbs) + "_" + str(rvar) + ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_object(vqe, filename)
    data.append(nqbs) 
    data.append(circuit.depth())
    
    print(filename)

    profiled_ip2_depol = get_profiled_state_depol(nqbs, tt, backend, False)
    tim = get_time(circuit, backend, 0)
    noise_sim = noise_simulation(circuit)
    dt = vqe._get_expectation_value_from_probs(noise_sim)
    esp = get_esp(circuit, noise_model, backend)
    if (esp < 0.10):
        return data
    idl = renormalize(ideal_simulation(circuit))
    ideal = vqe._get_expectation_value_from_probs(idl)
    
    extrapd2 = extrap_everything_depol(noise_sim, esp, profiled_ip2_depol)
    apr1 = extrap_everything_decoh_5(noise_sim, tim, nqbs)
    apr1_vec = extrap_everything_depol(apr1, esp, profiled_ip2_depol)
    apr1 = vqe._get_expectation_value_from_probs(apr1_vec)

    apr2 = extrap_everything_decoh_5_N(noise_sim, copy.deepcopy(circuit), 10)
    apr2_vec = extrap_everything_depol_N_r2(apr2, esp, profiled_ip2_depol, 10)
    apr2 = vqe._get_expectation_value_from_probs(apr2_vec)
    
    apr3_vec = extrap_everything_depol_N_r2(noise_sim, esp, profiled_ip2_depol, 10)
    apr3 = vqe._get_expectation_value_from_probs(apr3_vec)

    rzne_danly = vqe._get_expectation_value_from_probs(extrapd2)
    circuit.remove_final_measurements()
    dzne_extrp = get_DZNE_QAOA_extrap(copy.deepcopy(circuit), vqe)
    dzne_fstex, dzne_st = get_DZNE_QAOA_fstextr(copy.deepcopy(circuit), vqe)
    cdr_data = get_cdr_QAOA(copy.deepcopy(circuit), vqe)
    vncdr_data = get_vncdr_QAOA(copy.deepcopy(circuit), vqe)
    circuit.remove_final_measurements()
    
    qft = get_expectation_qraft(copy.deepcopy(circuit))
    qft_data = vqe._get_expectation_value_from_probs(qft)

    #data = data + str(circuit.count_ops()) 
    data.append(get_esp(circuit, noise_model, backend)) 
    data.append(tim)
    data.append(dt) 
    data.append(ideal) 
    data.append(rzne_danly)  
    data.append(apr3) 
    data.append(apr1)
    data.append(apr2)  
    data.append(dzne_extrp)
    data.append(dzne_fstex)
    data.append(cdr_data)
    data.append(vncdr_data)
    data.append(qft_data)
    data.append(get_absolute_error(ideal, dt))
    data.append(get_absolute_error(ideal, rzne_danly))
    data.append(get_absolute_error(ideal, apr3))
    data.append(get_absolute_error(ideal, apr1))
    data.append(get_absolute_error(ideal, apr2))
    data.append(get_absolute_error(ideal, dzne_extrp))
    data.append(get_absolute_error(ideal, dzne_fstex))
    data.append(get_absolute_error(ideal, cdr_data))
    data.append(get_absolute_error(ideal, vncdr_data))
    data.append(get_absolute_error(ideal, qft_data))
    data.append(get_fidelity(idl, noise_sim, nqbs))
    data.append(get_fidelity(idl, extrapd2, nqbs))
    data.append(get_fidelity(idl, apr3_vec, nqbs))
    data.append(get_fidelity(idl, apr1_vec, nqbs))
    data.append(get_fidelity(idl, apr2_vec, nqbs))
    data.append(get_fidelity(idl, dzne_st, nqbs))
    data.append(get_fidelity(idl, qft, nqbs))


    return data

def execute_GHZ(nqbs, rvar, dirname):
    data = []
    tt = 4
    vqe = supermarq.benchmarks.ghz.GHZ(nqbs)
    vqe_circuit = vqe.circuit()
    circuit = supermarq.converters.cirq_to_qiskit(vqe_circuit)
    circuit.remove_final_measurements()
    data.append(nqbs) 
    data.append(circuit.depth())
    filename = dirname + "/GHZ_" + str(nqbs) + "_" + str(rvar) + ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_object(vqe, filename)
    data.append(nqbs) 
    data.append(circuit.depth())
    
    print(filename)

    profiled_ip2_depol= get_profiled_state_depol(nqbs, tt, backend, False)

    noise_sim = noise_simulation(circuit)
    esp = get_esp(circuit, noise_model, backend)
    idl = renormalize(ideal_simulation(circuit))

    extrapd2 = extrap_everything_depol(noise_sim, esp, profiled_ip2_depol)
    apr1 = extrap_everything_decoh_5(noise_sim, get_time(circuit, backend, 0), nqbs)
    apr1_vec = extrap_everything_depol(apr1, esp, profiled_ip2_depol)
    
    apr2 = extrap_everything_decoh_5_N(noise_sim, copy.deepcopy(circuit), 10)
    apr2_vec = extrap_everything_depol_N_r2(apr2, esp, profiled_ip2_depol, 10)
    
    apr3_vec = extrap_everything_depol_N_r2(noise_sim, esp, profiled_ip2_depol, 10)
    
    circuit.remove_final_measurements()
    dzne_st = get_DZNE_extrapolated_state(circuit)
    circuit.remove_final_measurements()
    qft = get_expectation_qraft(copy.deepcopy(circuit))
    circuit.remove_final_measurements()
    
    data.append(get_esp(circuit, noise_model, backend)) 
    data.append(get_time(circuit, backend, 0))

    data.append(get_fidelity(idl, noise_sim, nqbs))
    data.append(get_fidelity(idl, extrapd2, nqbs))
    data.append(get_fidelity(idl, apr3_vec, nqbs))
    data.append(get_fidelity(idl, apr1_vec, nqbs))
    data.append(get_fidelity(idl, apr2_vec, nqbs))
    data.append(get_fidelity(idl, dzne_st, nqbs))
    data.append(get_fidelity(idl, qft, nqbs))
    
    
    return data

def execute_MerminBell(nqbs, rvar, dirname):
    data = []
    tt = 4
    vqe = supermarq.benchmarks.mermin_bell.MerminBell(nqbs)
    vqe_circuit = vqe.circuit()
    circuit = supermarq.converters.cirq_to_qiskit(vqe_circuit)
    circuit.remove_final_measurements()
    data.append(nqbs) 
    data.append(circuit.depth())
    filename = dirname + "/MerminBell_" + str(nqbs) + "_" + str(rvar) + ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_object(vqe, filename)
    data.append(nqbs) 
    data.append(circuit.depth())
    
    print(filename)

    profiled_ip2_depol= get_profiled_state_depol(nqbs, tt, backend, False)

    noise_sim = noise_simulation(circuit)
    esp = get_esp(circuit, noise_model, backend)
    idl = renormalize(ideal_simulation(circuit))

    extrapd2 = extrap_everything_depol(noise_sim, esp, profiled_ip2_depol)
    apr1 = extrap_everything_decoh_5(noise_sim, get_time(circuit, backend, 0), nqbs)
    apr1_vec = extrap_everything_depol(apr1, esp, profiled_ip2_depol)
    
    apr2 = extrap_everything_decoh_5_N(noise_sim, copy.deepcopy(circuit), 10)
    apr2_vec = extrap_everything_depol_N_r2(apr2, esp, profiled_ip2_depol, 10)
    
    apr3_vec = extrap_everything_depol_N_r2(noise_sim, esp, profiled_ip2_depol, 10)
    
    circuit.remove_final_measurements()
    dzne_st = get_DZNE_extrapolated_state(circuit)
    circuit.remove_final_measurements()
    qft = get_expectation_qraft(copy.deepcopy(circuit))
    circuit.remove_final_measurements()
    
    data.append(get_esp(circuit, noise_model, backend)) 
    data.append(get_time(circuit, backend, 0))

    data.append(get_fidelity(idl, noise_sim, nqbs))
    data.append(get_fidelity(idl, extrapd2, nqbs))
    data.append(get_fidelity(idl, apr3_vec, nqbs))
    data.append(get_fidelity(idl, apr1_vec, nqbs))
    data.append(get_fidelity(idl, apr2_vec, nqbs))
    data.append(get_fidelity(idl, dzne_st, nqbs))
    data.append(get_fidelity(idl, qft, nqbs))
    
    
    return data
