import numpy as np
from numpy.linalg import matrix_power
from math import sqrt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import Aer
import matplotlib.pyplot as plt

# Μέρος 1ο οριζω γεννήτορες και στόχο hadamard και μετρικές σφάλματος
phi = (1 + sqrt(5)) / 2
R00 = np.exp(-4j * np.pi / 5)  
R11 = np.exp( 3j * np.pi / 5)  
# ορισμος γεννήτορες σ1 σ2
sigma1 = np.array([[R00  ,     0],
                   [0    ,   R11]], dtype=complex)

sigma2 = np.array([[ (1/phi)*np.exp( 4j*np.pi/5)            , (1/np.sqrt(phi))*np.exp(-3j*np.pi/5)],
                   [ (1/np.sqrt(phi))*np.exp(-3j*np.pi/5)   ,                  -1/phi             ]],
                  dtype=complex)

# Hadamard στόχος
H = (1/np.sqrt(2)) * np.array([[1   ,   1],
                               [1   ,  -1]], dtype=complex)
# Hadamard braid word απο Monte Carlo
hadamard_word = [ ('s1', -1), ('s2',  1), ('s1',  -1), ('s2',  1),
    ('s1', -2), ('s2', -1), ('s1', 2), ('s2', -1), ('s1', 1),
    ('s2', -2), ('s1',  -2)]

#Ορισμός braidword U=σ1^a + σ2^b ...
def braid_unitary(word):
    U = np.eye(2, dtype=complex)
    gens = {'s1': sigma1, 's2': sigma2}
    for gen, k in word:
        U = U @ matrix_power(gens[gen], k) 
    return U

# ορισμός fidelity και τοπολογικής απόστασης d
def unitary_fidelity(U, V):
    d = 2
    X = U.conj().T @ V
    Fp = (abs(np.trace(X))**2) / (d*d)          # τυπος |Tr(U+V)|^2 / d^2
    Favg = (d*Fp + 1)/(d+1)                     # average fidelity
    return Fp, Favg

def d2_from_Favg(Favg):
    Fp = (3.0*Favg - 1.0)/2.0
    if Fp<0.0:
        Fp = 0.0
    elif Fp > 1.0:
        Fp=1.0
    return np.sqrt(1.0 - np.sqrt(Fp))

#Σύγκριση του H με το Braidword ως προς fidelity
U_braid = braid_unitary(hadamard_word)
Fp, Favg = unitary_fidelity(H, U_braid)
d_error = d2_from_Favg(Favg)
print(f"d error = {d_error :.6f}")
print(f"Average gate fidelity = {Favg:.6f}")


#Μερος 2ο qiskit και qasm
# 1) Οι πίνακες που προέκυψαν ορίζονται ως πύλες qiskit
Hgate      = UnitaryGate(H)
Hbraidgate = UnitaryGate(U_braid)

# 2) Μετρήσεις με qasm simulator
backend = Aer.get_backend('qasm_simulator')

# (a) ιδανικό H σε |0>
qcH = QuantumCircuit(1)
qcH.append(Hgate, [0])
qcH.measure_all()
jobH = backend.run(transpile(qcH, backend), shots=1000)
countsH = jobH.result().get_counts()
print("Counts H_ideal on |0> :", countsH)

# (b) braid H σε |0>
qcB = QuantumCircuit(1)
qcB.append(Hbraidgate, [0])
qcB.measure_all()
jobB = backend.run(transpile(qcB, backend), shots=1000)
countsB = jobB.result().get_counts()
print("Counts H_braid on |0> :", countsB)

# Plot histogram
from qiskit.visualization import plot_histogram
plot_histogram([countsH, countsB],
               legend=['H_ideal','H_braid'],
               title='QASM counts on |0>',)

plt.show()

#Μέρος 3ο εκτυπώσεις Πινάκων 
print("Braid Hadamard with Global Phase:")
print(np.round(U_braid, 3))

#Αφαίρεση Global Phase
# Βρίσκουμε τη φάση του στοιχείου [0,0]
phase_00 = np.angle(U_braid[0, 0])

# Αφαιρούμε τη φάση από όλο τον πίνακα
U_no_phase = U_braid * np.exp(-1j * phase_00)

print("\nBraid Hadamard no Global Phase:")
print(np.round(U_no_phase, 3))

# Σύγκριση με τον ιδανικό H
print("\nHadamard ideal:")
print(np.round(H, 3))

