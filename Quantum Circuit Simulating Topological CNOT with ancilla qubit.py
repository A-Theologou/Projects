import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import UnitaryGate
from qiskit.visualization import plot_histogram

# Μερός 1ο ορισμός βάσης και πυλών απο braid
phi = (1 + np.sqrt(5)) / 2
s1 = np.array([[np.exp(-4j*np.pi/5), 0], [0, np.exp(3j*np.pi/5)]], dtype=complex)
s2 = np.array([[(1/phi)*np.exp(4j*np.pi/5), (1/np.sqrt(phi))*np.exp(-3j*np.pi/5)],
               [(1/np.sqrt(phi))*np.exp(-3j*np.pi/5), -1/phi]], dtype=complex)

def remove_global_phase(U):
    det = np.linalg.det(U)
    U_norm = U / np.sqrt(det)
    if np.real(np.trace(U_norm)) < 0: U_norm *= -1
    return U_norm

def word_to_unitary(word):
    U = np.eye(2, dtype=complex)
    for gen, pwr in reversed(word):
        base = s1 if gen == 's1' else s2
        op = np.linalg.matrix_power(base, abs(pwr))
        if pwr < 0: op = np.linalg.inv(op)
        U = U @ op 
    return remove_global_phase(U)

# Δημιουργία των Braid Πυλών
U_H = word_to_unitary([('s1', -1), ('s2', 1), ('s1', -1), ('s2', 1), ('s1', -2),
                     ('s2', -1), ('s1', 2), ('s2', -1), ('s1', 1), ('s2', -2), ('s1', -2)])
U_X = word_to_unitary([('s1', 2), ('s2', 4), ('s1', -2), ('s2', -6),
                     ('s1', -2), ('s2', -5), ('s1', 2), ('s2', 6), ('s1', -1)])
U_T = word_to_unitary([('s2', 1), ('s1', 1), ('s2', -1), ('s2', -1), ('s1', -1), ('s2', 1), ('s1', 1),
                     ('s2', -1), ('s1', 1), ('s2', -1), ('s1', 1), ('s1', 1), ('s2', 1), ('s1', 1),
                       ('s2', 1), ('s1', -1), ('s2', 1), ('s1', -1), ('s1', -1), ('s2', 1)])
U_T_inv = np.linalg.inv(U_T)
U_Z = remove_global_phase(U_H @ U_X @ U_H)

# 2o μερος συναρτήσεις εκτέλεσης qiskit
#Ιδανική CNOT (Control σε |+>) χωρίς Ancilla
def run_ideal_cnot_pure():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    backend = AerSimulator()
    return backend.run(transpile(qc, backend), shots=2000).result().get_counts()

#Τοπολογική CNOT μέσω Injection (Hormozi Protocol)
def run_injection_protocol(with_correction=True):
    # 3 Qubits, 3 Classical bits (το bit 2 είναι της Ancilla)
    qc = QuantumCircuit(3, 3) 
    
    # Προετοιμασία
    qc.append(UnitaryGate(U_H, label="H_topo"), [0])
    qc.append(UnitaryGate(U_H, label="H_topo"), [2])
    qc.cx(2, 1) # Bell pair creation
    
    # Injection Weave
    qc.append(UnitaryGate(U_T, label="Inj_T"), [2])
    qc.cx(0, 2)
    qc.append(UnitaryGate(U_T_inv, label="Inv_T"), [2])
    
    # Μέτρηση Ancilla
    qc.measure(2, 2)
    
    if with_correction:
        # adaptive feedback
        with qc.if_test((qc.clbits[2], 1)):
            qc.append(UnitaryGate(U_X, label="X_corr"), [1])
            qc.append(UnitaryGate(U_Z, label="Z_corr"), [1])
            
    qc.measure(0, 0) # Control
    qc.measure(1, 1) # Target
    
    backend = AerSimulator()
    return backend.run(transpile(qc, backend), shots=4000).result().get_counts()

def split_counts_by_ancilla(counts):
    anc0, anc1 = {}, {}
    for state, val in counts.items():
        #  ancilla είανι το  q2 
        anc_bit = state[0]
        pure_state = state[1:] 
        if anc_bit == '0':
            anc0[pure_state] = val
        else:
            anc1[pure_state] = val
    return anc0, anc1

def ensure_all_states(counts_dict):
    all_possible = ['00', '01', '10', '11']
    return {state: counts_dict.get(state, 0) for state in all_possible}

# 3o μέρος εκτέλεση και γραφήματα
counts_ideal = run_ideal_cnot_pure()
counts_with = run_injection_protocol(with_correction=True)
counts_without = run_injection_protocol(with_correction=False)

with_anc0, with_anc1 = split_counts_by_ancilla(counts_with)
without_anc0, without_anc1 = split_counts_by_ancilla(counts_without)

# Σχεδιασμός διαγραμμάτων
plot_data_anc0 = [ensure_all_states(counts_ideal), ensure_all_states(with_anc0), ensure_all_states(without_anc0)]
plot_data_anc1 = [ensure_all_states(counts_ideal), ensure_all_states(with_anc1), ensure_all_states(without_anc1)]

# Γράφημα 1 Ancilla = 0
fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_subplot(111)
plot_histogram(plot_data_anc0, legend=['Ideal CNOT', 'Braided (With If)', 'Braided (No If)'], ax=ax1)
ax1.set_title('Case 1: Ancilla Measurement = 0 (Natural Success)')

# Γράφημα 2 Ancilla = 1
fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(111)
plot_histogram(plot_data_anc1, legend=['Ideal CNOT', 'Braided (With If)', 'Braided (No If)'], ax=ax2)
ax2.set_title('Case 2: Ancilla Measurement = 1 (Correction Needed)')

# 4ο μέρος σχεδίαση κυκλώματοςΣχεδίασε κυκλώματος 
qc_draw = QuantumCircuit(3, 3) # q0:Control, q1:Target, q2:Ancilla

# Βήμα 1 Entanglement
qc_draw.append(UnitaryGate(U_H, label="H"), [0])
qc_draw.append(UnitaryGate(U_H, label="H"), [2])
qc_draw.cx(2, 1) 
qc_draw.barrier(label="Entangle")

# Βήμα 2 Injection 
qc_draw.append(UnitaryGate(U_T, label="Inj_T"), [2])
qc_draw.cx(0, 2)
qc_draw.append(UnitaryGate(U_T_inv, label="Inv_T"), [2])
qc_draw.barrier(label="Inject")

# Βήμα 3 Measure & Correct if
qc_draw.measure(2, 2)
with qc_draw.if_test((qc_draw.clbits[2], 1)):
    qc_draw.append(UnitaryGate(U_X, label="X"), [1])
    qc_draw.append(UnitaryGate(U_Z, label="Z"), [1])

qc_draw.measure([0, 1], [0, 1])

# Εμφάνιση
qc_draw.draw('mpl', style='iqp') 

plt.show()