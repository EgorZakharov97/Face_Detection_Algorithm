import numpy as np
from numpy import linalg as la
import math
import imagefuncs_A3 as imf
import os

SIZE = 200
MAX_SHADE = 255
BORDER = 20
PATH = 'Faces/'

def readFaces():
    file_names = os.listdir(PATH)
    d = np.zeros((SIZE**2, len(file_names)), dtype=np.int32)

    for i, filename in enumerate(file_names, start=0):
        img = imf.read_image(PATH + filename)
        max_shade, data = img
        reshaped = data.reshape(-1)
        d[:,1] = reshaped

    return d
    

def step1():
    print("Reading faces...")
    d = readFaces()
    # d = np.array([[1,2,3],[4,5,6],[7,8,9]])
    
    print("Pricessing...")
    # Calculate the average column x
    mean = d.mean(axis=1)
    # Subtract x from every column of the d x n matrix
    L = (d.transpose() - mean).transpose()
 




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

    return np.array(eigenFaces)

        

#STEP 3 
#Finds weight vector for column Lj of matrix j
def findWeight(eigenFaces, Lj):
    weightVector = [0] * len(eigenFaces)
    xj = Lj.flatten()

    for i in range(0, len(eigenFaces)):
        #get column slice from eigenFaces and flatten 
        vi = eigenFaces[i].flatten()
       
        weightVector[i] = np.dot( xj, vi )

    return weightVector



step1()
