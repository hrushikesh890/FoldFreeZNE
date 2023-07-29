import random, pickle, os, copy, random
from cirq import NoiseModel
from helper_functions.rzne import extrap_everything_depol, get_profiled_state_depol, get_esp, renormalize
from qiskit import QuantumCircuit, execute, Aer
from qiskit.providers import aer
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import psutil

from helper_functions.conversions import dict_to_array


def scrambled(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest


def read_dict(filename):
    if os.path.isfile(filename):
        f = open(filename, "rb")
        file_content = {}
        while 1:
            try:
                file_content.update(pickle.load(f))
            except (EOFError):
                break
        f.close()
    else:
        file_content = {}
    return file_content


def apply_measurement(circuit, qubits):
    measured_circuit = QuantumCircuit(circuit.num_qubits, len(qubits))
    for circuit_inst, circuit_qubits, circuit_clbits in circuit.data:
        measured_circuit.append(circuit_inst, circuit_qubits, circuit_clbits)
    measured_circuit.barrier(qubits)
    measured_circuit.measure(qubits, measured_circuit.clbits)
    return measured_circuit


def find_process_jobs(jobs, rank, num_workers):
    count = int(len(jobs) / num_workers)
    remainder = len(jobs) % num_workers
    if rank < remainder:
        jobs_start = rank * (count + 1)
        jobs_stop = jobs_start + count + 1
    else:
        jobs_start = rank * count + remainder
        jobs_stop = jobs_start + (count - 1) + 1
    process_jobs = list(jobs[jobs_start:jobs_stop])
    return process_jobs


def evaluate_circ(circuit, backend, options=None):
    circuit = copy.deepcopy(circuit)
    max_memory_mb = psutil.virtual_memory().total >> 20
    max_memory_mb = int(max_memory_mb / 4 * 3)
    if backend == "statevector_simulator":
        simulator = aer.Aer.get_backend("statevector_simulator")
        result = simulator.run(circuit).result()
        statevector = result.get_statevector(circuit)
        prob_vector = Statevector(statevector).probabilities()
        return prob_vector
    elif backend == "noiseless_qasm_simulator":
        simulator = aer.Aer.get_backend("aer_simulator", max_memory_mb=max_memory_mb)
        if isinstance(options, dict) and "num_shots" in options and options["num_shots"] is not None:
            num_shots = options["num_shots"]
        else:
            num_shots = max(32768, 2**circuit.num_qubits)

        if isinstance(options, dict) and "memory" in options:
            memory = options["memory"]
        else:
            memory = False
        if circuit.num_clbits == 0:
            circuit.measure_all()
        print(num_shots)
        result = simulator.run(circuit, shots=num_shots, memory=memory).result()

        if memory:
            qasm_memory = np.array(result.get_memory(circuit))
            assert len(qasm_memory) == num_shots
            return qasm_memory
        else:
            noiseless_counts = result.get_counts(circuit)
            assert sum(noiseless_counts.values()) == num_shots
            noiseless_counts = dict_to_array(
                distribution_dict=noiseless_counts, force_prob=True
            )
            return noiseless_counts

    elif backend == "qasm_simulator":
        if isinstance(options, dict) and "noise_model" in options:
            nem = options["noise_model"]
        else:
            nem = None
        if isinstance(options, dict) and "backend_config" in options:
            backend_config = options["backend_config"]
            coupling_map = backend_config.configuration().coupling_map
            basis_gates = nem.basis_gates
        else:
            backend_config = None

        if isinstance(options, dict) and "num_shots" in options and options["num_shots"] is not None:
            num_shots = options["num_shots"]
        else:
            num_shots = 32768

        if isinstance(options, dict) and "memory" in options:
            memory = options["memory"]
        else:
            memory = False
        
       
        if circuit.num_clbits == 0:
            circuit.measure_all()
        #print(num_shots)
        result = execute(circuit, Aer.get_backend('qasm_simulator'), coupling_map=coupling_map, basis_gates=basis_gates, noise_model=nem, shots = 32768, optimization_level=0).result()

        if memory:
            qasm_memory = np.array(result.get_memory(circuit))
            assert len(qasm_memory) == num_shots
            return qasm_memory
        else:
            noiseless_counts = result.get_counts(circuit)
            if isinstance(options, dict) and "mitigated" in options:
                if options["mitigated"]:
                    profile = get_profiled_state_depol(circuit.num_qubits, 10, backend_config, False)
                    noiseless_counts = renormalize(noiseless_counts)
                    print("esp ", get_esp(circuit, nem, backend_config), circuit.num_qubits)
                    noiseless_counts = extrap_everything_depol(noiseless_counts, get_esp(circuit, nem, backend_config), profile)
                    
            #print(noiseless_counts)
            #assert sum(noiseless_counts.values()) == num_shots
            noiseless_counts = dict_to_array(
                distribution_dict=noiseless_counts, force_prob=True
            )
            return noiseless_counts
    else:
        raise NotImplementedError


def circuit_stripping(circuit):
    # Remove all single qubit gates and barriers in the circuit
    dag = circuit_to_dag(circuit)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circuit.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) == 2 and vertex.op.name != "barrier":
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)


def dag_stripping(dag, max_gates):
    """
    Remove all single qubit gates and barriers in the DAG
    Only leaves the first max_gates gates
    If max_gates is None, do all gates
    """
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(dag.qregs[qreg_name]) for qreg_name in dag.qregs]
    vertex_added = 0
    for vertex in dag.topological_op_nodes():
        within_gate_count = max_gates is None or vertex_added < max_gates
        if vertex.op.name != "barrier" and len(vertex.qargs) == 2 and within_gate_count:
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
            vertex_added += 1
    return stripped_dag
