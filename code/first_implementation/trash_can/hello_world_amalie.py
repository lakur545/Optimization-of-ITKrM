import numpy as np
import ImportImages

np.random.seed(1)

def find_max_indices(D,y,S,K):
    all_values = np.zeros(K)
    for i in range(0,K):        # D.T@y gives same result as the loop
        all_values[i] = np.abs(D[:,i]@y)
    index = np.argpartition(all_values, -S)[-S:]
    return index


def update_dictionary(D,y,I_Dn, k):
    d_k = 0
    for n in range(0,N):
        if np.any(np.isin(I_Dn[:,n], k)):
            d_k = d_k + (y[:,n] - P(D[:,I_Dn[:,n]])@y[:,n] + np.outer(D[:,k], D[:,k])@y[:,n])

    return d_k

def P(A): #Calculate projection
    P = A@np.linalg.pinv(A)
    return P


K = 128      # Number of columns in D (Atoms)
M = 16      # Number of rows in D (Number of pixels in training image)
N = 1000    # Number of training examples
S = 3       # Number of used vectors (Sparsity). Amount that is NOT zero.

data = np.load('grayScale32x32cars.npy')

W_data = 32    # Width in pixels
H_data = 32    # Height in pixels
N_subpic = 4     # Width/Height in pixels of smaller square extracted from image.

smallSet = ImportImages.ExtractSmall(data, W_data, H_data, N_subpic)
newPic = ImportImages.MergeSmall(smallSet, W_data, H_data, N_subpic)

Ix = np.random.choice(K, size=(N, S))   # Random number drawn from pool.
x = np.zeros((K, N))
for i in range(N):
    x[Ix[i, :], i] = 1

D_init = np.random.rand(M, K)
for k in range(0,K):
    D_init[:,k] = D_init[:,k]/np.linalg.norm(D_init[:,k])  # Normalise vectors of initial dictionary.
y = D_init @ x
y = smallSet.T[:,:N]

D = y[:,:K]
for k in range(0,K):
    D[:,k] = D[:,k]/np.linalg.norm(D[:,k])  # Normalise vectors of initial dictionary.

T = 20

I_Dn = np.zeros((S, N))

#print('D: \n',D,'\n')
#print('y:\n',y)
for t in range(0,T):
    print(t, 'i')
    for n in range(0,N):
        I_Dn[:,n] = find_max_indices(D, y[:,n], S, K)

    I_Dn = I_Dn.astype(int)
    D_temp = np.zeros((M, K))
    print(t, 'k')
    for k in range(0,K):
        D_temp[:,k] = update_dictionary(D, y, I_Dn, k)
        D_temp[:,k] = D_temp[:,k]/np.linalg.norm(D_temp[:,k])
    D = D_temp
    print(np.linalg.norm(D-D_init))
#print('I_Dn \n',I_Dn)
#print('D \n', D)
