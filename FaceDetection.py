import numpy as np
from numpy import linalg as la
import math
import imagefuncs_A3 as imf
import os
import sys

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
        d[:,i] = reshaped

    return d

def detectFace(filename):
    print("Reading faces...")
    d = readFaces()
    n = d.shape[1]
    
    print("Building library...")

    # Calculate the average column x
    mean = d.mean(axis=1)

    # Subtract x from every column of the d x n matrix
    # as a result we get a transpose of L
    LT = (d.transpose() - mean)

    # find L
    L = LT.transpose()

    # find LTL by matrix multiplication
    LTL = np.matmul(LT, L)

    # divide LTL by (n-1)
    multiplier = 1/(n-1)
    LTL = multiplier * LTL

    # find eigenfaces
    eigenfaces = findEigenFaces(LTL, L)

    # find weights
    weights = [0] * n

    for i in range(n):
        col_L = L[:,i]
        weights[i] = findWeight(eigenfaces, col_L)

    weights = np.array(weights)

    # Test an image
    print("Testing...\n")
    testImage(eigenfaces, weights, mean, filename)
    

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

    return np.array(weightVector)

# STEP 4
# Read a test image, concatenate pixels
# Test if the image is a face
def testImage(eigenfaces, weights, mean, filename="test.png"):
    img = imf.read_image(filename)
    max_shade, data = img
    data = data.flatten()

    z = data - mean # subtract mean from image data

    w = findWeight(eigenfaces, z)

    distances = [0] * len(weights) # the distance vector

    for i in range(len(weights)):
        distances[i] = abs(weights[i] - w)
        index = i

    d = np.min(distances) # the minimal distance to a pic from library

    print("The closest distance is " + str(d))


file_to_test = "./Faces/face_1.pgm"

if len(sys.argv) > 1:
    file_to_test = sys.argv[1]

detectFace(file_to_test)