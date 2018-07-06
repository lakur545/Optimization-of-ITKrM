import numpy as np

np.random.seed(0)


def find_max_indices(D,y,S,K):
    all_values = np.zeros(K)
    for i in range(0,K):
        all_values[i] = np.abs(D[:,i]@y) 
    index = np.argpartition(all_values, -S)[-S:] # Checking which atoms of D that maximizes the optimization problem

    return index


def update_dictionary(D,y,I_Dn, k): # calculate atoms
    d_k = 0
    for n in range(0,N):
        if np.any(np.isin(I_Dn[:,n], k)): # For only using the active atoms of the dictionary 
            d_k += np.sign(D[:,k]@y[:,k])*(y[:,n] - P(D[:,I_Dn[:,n]])@y[:,n] + np.outer(D[:,k], D[:,k])@y[:,n]) # P(d_k) is calculated as the outer product of d_k and d_k? 

#    print('d_k \n', d_k)
    return d_k/N

def P(A): #Calculate projection
    P = A@np.linalg.pinv(A)

    return P

K = 5       # Number of columns in D (Atoms)
M = 5       # Number of rows in D
S = 3       # Number of used vectors (Sparsity)
N = 100       # Number of training examples
T = 100
y = np.random.randint(5, size = (M,N)) # N vectors with M rows
D = np.random.randint(5, size = (M,K)) # random initial Dictionary matrix

for i in range(1,M+1):
    y[i,:] = y[i,:]*i

I_Dn = np.zeros((S,N))

print('D: \n',D,'\n')
print('y:\n',y)
for t in range(0,T):
    for i in range(0,N):
        I_Dn[:,i] = find_max_indices(D,y[:,i],S,K) 
    
    I_Dn = I_Dn.astype(int)
    D_temp = np.zeros((M,K)) # Initialize temporary dictionary
    for k in range(0,K):
        D_temp[:,k] = update_dictionary(D,y,I_Dn, k) #Calculating  atoms
        D_temp[:,k] = D_temp[:,k]/np.linalg.norm(D_temp[:,k]) # Normalizing atoms 
    D = D_temp
print('I_Dn \n',I_Dn)
print('D \n', D)
