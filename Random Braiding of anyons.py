import numpy as np                     #numpy λειτουργιες
from numpy.linalg import matrix_power  #να μπορω να βάζω δυνάμεις στους πίνακες
import random                          #να μπορω να παρω random τιμες
import matplotlib.pyplot as plt        #διαγραμματα



#γεννήτορες του braid word σ1 σ2 με βαση τα F και R moves
phi = (1 + np.sqrt(5)) / 2             #χρυσός λόγος 
R00 = np.exp(-4j * np.pi / 5)          #R moves 
R11 = np.exp( 3j * np.pi / 5)
sigma1 = np.array([[R00, 0],                  
                   [0,   R11]], dtype=complex)

sigma2 = np.array([[ (1/phi)*np.exp( 4j*np.pi/5), (1/np.sqrt(phi))*np.exp(-3j*np.pi/5)],
                   [ (1/np.sqrt(phi))*np.exp(-3j*np.pi/5),           -1/phi           ]],
                  dtype=complex)



#Ιδανικές πύλες που προσσεγίζω Η,Τ,Χ
# στόχος Hadamard
H = (1/np.sqrt(2)) * np.array([[1, 1],
                               [1,-1]], dtype=complex)
#στοχος Τ ή π/4
T = np.array([[1     ,          0        ],
              [0     ,np.exp(1j* np.pi/4)]], dtype=complex)
#στοχος X (απαραιτητη για CNOT)
X_gate = np.array([[0, 1],
                   [1, 0]])


#Μετρικές σφάλματος,  πιστότητα και error d2(για διαγραμμα σφαλμα/μηκος)
def Favg_unitaries(U, V):
    d = 2
    X = U.conj().T @ V
    Fp = (abs(np.trace(X))**2) / (d*d)
    return (d*Fp + 1)/(d+1)
def d2_from_Favg(Favg):
    Fp = (3.0*Favg - 1.0)/2.0
    if Fp<0.0:
        Fp = 0.0
    elif Fp > 1.0:
        Fp=1.0
    return np.sqrt(1.0 - np.sqrt(Fp))


#Βηματα
#υπολογίζω τους αντίστροφους ώστε να γλυτώσω πραξεις
S1   = sigma1
S1m1 = matrix_power(sigma1, -1)
S2   = sigma2
S2m1 = matrix_power(sigma2, -1)
#λεξικο με τους γεννητορες μου
STEP_MAT = {
    ('s1', +1): S1,
    ('s1', -1): S1m1,
    ('s2', +1): S2,
    ('s2', -1): S2m1,
}
#προσθηκη επομενου βήματος στο braidword
def apply_step(U, action):         #οπου action ενα tuple πχ ('s1', +1)
    return U @ STEP_MAT[action] 

#random δημιουργια της braidword

Runs = 100000            #θα εχω 10000 φορες να τρεξει για μηκος 1 , 100000 φορες για μηκος 2 κτλ
Max_steps = 30         #συνολικα ποσο μηκος θα εχω
Actions = [('s1', +1), ('s1', -1), ('s2', +1), ('s2', -1)]    

best_F_per_len = np.full(Max_steps, 0.0 , dtype= float)
best_word_per_len= [None] * Max_steps

for r in range(1, Runs + 1):
    U = np.eye(2, dtype=complex)
    word_steps = []
    steps_used = 0
    last = None 
    
    while steps_used < Max_steps:
        gen, power = random.choice(Actions)


        if last is not None and last[0] == gen and last [1] == -power:
            continue

        action = (gen, power)
        U = apply_step(U, action)
        word_steps.append(action)
        last = action
        steps_used += 1


        F_cur = Favg_unitaries(U, T)
        idx = steps_used - 1

        if F_cur > best_F_per_len[idx]:
            best_F_per_len[idx] = F_cur
            best_word_per_len[idx] = word_steps[:]

#μηκος για αξονα χ
lengths = np.arange(1, Max_steps+1)

#d2 απο τα καλυτερα Fidelity καθε μήκους για αξονα y
d2_vals = np.array([d2_from_Favg(f) for f in best_F_per_len])

#εκτύπωση αποτελεσμάτων
print("Best braidword per length:")
for L in [1, 2, 3, 5, 10, 20, Max_steps]:
    print(f"L={L:3d} best Favg={best_F_per_len[L-1]:.6f}")
    print(f"best word = {best_word_per_len[L-1]}")

#διαγραμμα
plt.semilogy(lengths, d2_vals,marker='o')
plt.xlabel('length')
plt.ylabel('d2')
plt.grid(True,which='both')
plt.show()



