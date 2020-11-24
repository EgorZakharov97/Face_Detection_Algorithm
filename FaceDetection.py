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
        d[:,1] = reshaped

    return d

def detectFace(filename):
    print("Reading faces...")
    d = readFaces()
    # d = np.array([[1,2,3],[4,5,6],[7,8,9]])
    
    print("Building library...")
    # Calculate the average column x
    mean = d.mean(axis=1)
    # Subtract x from every column of the d x n matrix
    LT = (d.transpose() - mean)
    L = LT.transpose()

    covMatrix = np.matmul(LT, L)
    covMatrix = (1/(SIZE-1))*covMatrix

    eigenFaces = findEigenFaces(covMatrix, L)

    weights = np.zeros((SIZE**2,))

    for i in range(SIZE**2):
        weights[i] = findWeight(eigenFaces, L[:i])

    print("Testing...\n")
    is_face = testImage(weights, mean, filename)

    # the test image is the image we already have in the library
    if(is_face > 0):
        print("The image is a face that we already have")
    elif(is_face == 0):
        print("The image is a new face")
    else:
        print("The image is not a face")

    print()
    

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

# STEP 4
# Read a test image, concatenate pixels
# Test if the image is a face
def testImage(weights, mean, filename="test.png"):
    img = imf.read_image(filename)
    max_shade, data = img
    data = data.flatten()

    mean = np.mean(mean) # mean is an array, we want an int
    z = data - mean # subtract mean from image data

    # calculate the covariance matrix
    covMatrix = np.matmul(z.transpose(), z)
    covMatrix = (1/(SIZE-1))*covMatrix

    eigenFace = findEigenFaces(covMatrix, z) # find eigenFaces
    w = findWeight(eigenFace, z) # z is a 1d array, we can pass it as it is

    distances = np.zeros((SIZE)) # the distance vector

    for i in range(weights):
        distances[i] = math.abs(weights[i] - w)

    d = np.min(distances) # the minimal distance to a pic from library

    # low and high boundaries
    d_low = np.min(weights)
    d_high = np.max(weights)

    # the test image is the image we already have in the library
    if(d < d_low):
        return 1
    # the image is a new face
    elif(d > d_high):
        return 0
    # the image is not a face
    else:
        return -1


file_to_test = "Faces/face_1.png"

if len(sys.argv) > 1:
    file_to_test = sys.argv[1]

detectFace(file_to_test)