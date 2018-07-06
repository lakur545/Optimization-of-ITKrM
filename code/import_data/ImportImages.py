import numpy as np
import matplotlib.pyplot as plt

data = np.load('grayScale32x32cars.npy')

W=32    # Width in pixels
H=32    # Height in pixels
N=4     # Width/Height in pixels of smaller square extracted from image.

pictureNumber = 0

plt.figure(0, dpi=50)
plt.imshow(data[pictureNumber,:].reshape(32,32), cmap='gray', vmin=0, vmax=255)

def ExtractSmall(data, W, H, N):
    """
    Extracts small NxN pictures from data set of WxH pictures and save in new data set
    that now holds more training examples. Pictures in data matrix must be saved with
    each row of an image appended to form a vector of length W*H and each picture is
    a new row in data matrix.

    Creates a new set of training examples of NxN squares extracted from the pictures.
    W/N and H/N must be integers.
    """
    pic = np.zeros((W,H))
    newSet = np.zeros((len(data[:,0])*W*H//(N*N), N*N))     # Create an empty data set that holds data from all NxN squares.
    for k in range(0, len(data[:,0])):
        pic = data[k,:].reshape(W,H)    # Extract the k-th picture from the data set
        for i in range(0, H//N):
            for j in range(0, W//N):
                newSet[k*W*H//(N*N)+j+i*W//N,:] = pic[i*N:(i+1)*N, j*N:(j+1)*N].reshape(N*N)    # Extract the (i,j)-th NxN square from the k-th picture.
    return newSet

def MergeSmall(newSet, W, H, N):
    """
    Merge the NxN pictures into data set with pictures of size WxH.
    Returns a 3d data set of size [pictureNumber, H, W].
    """
    data_len=len(newSet)//((H*W)//(N*N))
    newPic = np.zeros((data_len,H,W))     # Create data set of blank pictures with size WxH
    for k in range(0, data_len):
        for i in range(0, H//N):
            for j in range(0, W//N):
                newPic[k, i*N:(i+1)*N, j*N:(j+1)*N] = newSet[j+i*H//N+k*W*H//(N*N),:].reshape(N,N)  # Add (i,j)-th NxN square to k-th picture.
    return newPic

#plt.imshow(newSet[64,:].reshape(N,N), cmap='gray', vmin=0, vmax=255)

smallSet = ExtractSmall(data, W, H, N)  # Create a set of small NxN squares from data set
newPic = MergeSmall(smallSet, W, H, N)  # Create 3d matrix of images from set of small NxN squares

plt.figure(1, dpi=50)
plt.imshow(newPic[pictureNumber,:,:], cmap='gray', vmin=0, vmax=255)
