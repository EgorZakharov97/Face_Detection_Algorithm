import numpy as np
from numpy import linalg as la
import math
import imagefuncs_A3 as imf
import os
import sys

#Step 2
#find eigenvalues, eigenvectors, and corresponding eigenfaces
def findEigenFaces(covMatrix, L):

    #find eigenvalues and eigenvectors and sort them in ascending order
    eigenValues, eigenVectors = la.eig(covMatrix)

    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    # #find k eigenvalues such that they sum to 9k% of the total
    eigenValSum = 0.95 * np.sum(eigenValues )
    sum = 0
    k = 0
    while( sum < eigenValSum and eigenValues[k]+sum < eigenValSum):
        sum += eigenValues[k]
        k = k + 1

    #edge case for when the sum is less than the largest eigenvalue 
    if(k == 0):
        eigenValues = eigenValues[0:1]
        eigenVectors = eigenVectors[:, 0:1]

    else:
        eigenValues = eigenValues[0:k]
        eigenVectors = eigenVectors[:, 0:k]

    #find eigenfaces
    eigenFaces = [] 

    for colNum in range(0, len(eigenVectors[0])):
        temp = eigenVectors[:, colNum:colNum+1]
        temp = np.matmul(L, temp)
        temp = temp / la.norm(temp)
        eigenFaces.append( temp )

    return np.array(eigenFaces).flatten()

M = np.array(
    [
        [1,2,3],
        [4,5,6],
        [0,1,0]
    ]
)

print(M[:,0])