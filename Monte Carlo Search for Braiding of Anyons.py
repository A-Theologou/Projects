import numpy as np
from numpy.linalg import matrix_power
import random
import math
import matplotlib.pyplot as plt

#Μερος 1 ορισμός γεννητορων για fibonacci anyons
#γενήτορες σ1:πλέξη πρώτου με δεύτερου anyon σ2:πλέξη δεύτερου με τρίτου
phi = (1 + np.sqrt(5)) / 2
R00 = np.exp(-4j * np.pi / 5)
R11 = np.exp( 3j * np.pi / 5)

sigma1 = np.array([[R00, 0],
                   [0,   R11]], dtype=complex)

sigma2 = np.array([[ (1/phi)*np.exp( 4j*np.pi/5), (1/np.sqrt(phi))*np.exp(-3j*np.pi/5)],
                   [ (1/np.sqrt(phi))*np.exp(-3j*np.pi/5),           -1/phi           ]],
                  dtype=complex)


#Μερος 2 οι πύλες στόχοι και οι μετρικές σφάλματος
#στόχος Hadamard
H_gate = (1/np.sqrt(2)) * np.array([[1, 1],
                               [1,-1]], dtype=complex)
#στόχος Τ 
T_gate = np.array([[1     ,          0        ],
              [0     ,np.exp(1j* np.pi/4)]], dtype=complex)
#στόχος X 
X_gate = np.array([[0, 1],
                   [1, 0]])

#μετρικές σφάλματος 
#Favg
def Favg_unitaries(U, V):
    d = 2
    X = U.conj().T @ V
    Fp = (abs(np.trace(X))**2) / (d*d)
    return (d*Fp + 1)/(d+1)

#d2 τοπολογική απόσταση από Favg που αποτελεί τη συνάρτηση κόστους του monte carlo
def d2_from_Favg(Favg):
    inside = (3.0*Favg - 1.0)/2.0
    if inside < 0:
        inside = 0.0
    return np.sqrt(1.0 - np.sqrt(inside))

#Μερος 3 Προετοιμασία κινήσεων Move Set
#προυπολογισμός των δυνάμεων για ταχύτητα
S1   = sigma1
S1m1 = matrix_power(sigma1, -1)
S2   = sigma2
S2m1 = matrix_power(sigma2, -1)

STEP_MAT = {
    ('s1', +1): S1,
    ('s1', -1): S1m1,
    ('s2', +1): S2,
    ('s2', -1): S2m1,
}
ACTIONS = [('s1', +1), ('s1', -1), ('s2', +1), ('s2', -1)]

def apply_step(U, action):
    g, s = action
    return U @ STEP_MAT[(g, s)]

def U_from_word(word):
    U = np.eye(2, dtype=complex)
    for a in word:
        U = apply_step(U, a)
    return U

def inverse_of(a):  
    return (a[0], -a[1])

#αποφεύγονται άμεσες ακυρώσεις πχ σ^1 και μετα να ακολουθεί σ^-1 
def valid_to_place(word, i, a):
    if i > 0 and word[i-1] == inverse_of(a):
        return False
    if i < len(word)-1 and word[i+1] == inverse_of(a):
        return False
    return True

#Μερος 4 αλγόριθμος monte carlo
#τυχαία αρχική λέξη αποφεύγοντας ακυρώσεις
def random_init_word(L, rng):
    w = []
    for i in range(L):
        choices = [a for a in ACTIONS if valid_to_place(w, i, a)]
        w.append(rng.choice(choices))
    return w

# αλλαγή ενός τυχαίου βήματος της πλέξης 
def mutate_word(word, rng):
    L = len(word)
    i = rng.randrange(L)
    current = word[i]
    #  κινήσεις όλες εκτός της ίδιας
    cand_actions = [a for a in ACTIONS if a != current]
    rng.shuffle(cand_actions)
    for a in cand_actions:
        if valid_to_place(word, i, a):
            new_word = word[:]
            new_word[i] = a
            return new_word
    # αν δεν βρέθηκε  επιστρέφει το ίδιο
    return word[:]

#παράμετροι monte calro
MAX_LEN = 30            # θα σαρώσουμε μήκη 1 - MAX_LEN
TRIES_PER_LEN = 10000    # δοκιμές ανα μήκος 
T0, T1 = 0.3, 1e-3      # αρχική/τελική θερμοκρασία
rng = random.Random(None)

best_F_per_len = np.full(MAX_LEN, -1.0, dtype=float)
best_word_per_len = [None for _ in range(MAX_LEN)]

for L in range(1, MAX_LEN+1):
    # αρχικοποίηση λέξης μήκους L
    cur = random_init_word(L, rng)
    curF = Favg_unitaries(U_from_word(cur), T_gate)
    best, bestF = cur[:], curF

    for t in range(1, TRIES_PER_LEN+1):
        T = T0 * (T1/T0)**(t/TRIES_PER_LEN)    #υπολογισμός θερμοκρασίας

        cand = mutate_word(cur, rng)
        Fc = Favg_unitaries(U_from_word(cand), T_gate)
        d = Fc - curF  # μεγιστοποιούμε το Favg
        #κριτήριο αποδοχής
        if d >= 0 or rng.random() < math.exp(d/max(T,1e-12)):
            cur, curF = cand, Fc
            if curF > bestF:
                best, bestF = cur[:], curF

    best_F_per_len[L-1] = bestF
    best_word_per_len[L-1] = best
    print(f"L={L:2d}  best Favg={bestF:.6f}")

#Μερος 5ο αποτελέσματα και γραφήματα
lengths = np.arange(1, MAX_LEN+1)
best_so_far_F = np.maximum.accumulate(best_F_per_len)
best_so_far_d2 = np.array([d2_from_Favg(f) for f in best_so_far_F])

#γράφημα
plt.figure(figsize=(6,4))
plt.semilogy(lengths, best_so_far_d2, marker='o', linewidth=1)
plt.xlabel('length')
plt.ylabel('d2')
plt.grid(True, which='both', linewidth=0.5)
plt.tight_layout()
plt.savefig('sa_best_error_vs_length.png', dpi=200)
plt.show()
print("\nSaved figure as: sa_best_error_vs_length.png")

#Εκτύπωση του καλύτερου αποτελέσματος
best_index = np.argmax(best_F_per_len)
overall_best_Favg = best_F_per_len[best_index]
overall_best_word = best_word_per_len[best_index]
overall_best_d2 = d2_from_Favg(overall_best_Favg)

print(f"Best Length= {best_index + 1}")
print(f"Best Favg= {overall_best_Favg:.8f}")
print(f"Least d2= {overall_best_d2:.8f}")
print("Braid Word:")
print(overall_best_word)
