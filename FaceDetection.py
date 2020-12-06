import numpy as np
from numpy import linalg as la
import math
import imagefuncs_A4 as imf
import os
import sys

PATH_TO_FACES = 'Faces/'
PATH_TO_DATA = 'data/'
EIGEN_FACES = 'eigenfaces'
WEIGHTS = 'weights'
MEAN = 'mean'
FILE_NAMES = 'file_names'
SIZE = 200


sigma = 2
r = 4

gauss2 = imf.gaussian1D(sigma, r)
eigenfaces = np.array([])
weights = np.array([])
mean = np.array([])
file_names = []

def getData():
    global eigenfaces
    global weights
    global mean
    global file_names

    eigenfaces = np.load(PATH_TO_DATA + EIGEN_FACES + '.npy')
    weights = np.load(PATH_TO_DATA + WEIGHTS + '.npy')
    mean = np.load(PATH_TO_DATA + MEAN + '.npy')
    file_names = np.load(PATH_TO_DATA + FILE_NAMES + '.npy')

def detectEdges(image):
    smooth_image = imf.convolve2D_separable(image, gauss2)
    gradientjk = imf.gradient_vectors(smooth_image.data)
    grad_lengths = imf.gradient_lengths(gradientjk)
    grad_max = np.amax(grad_lengths)
    grad_image = imf.PGMFile(grad_max, grad_lengths)
    thin_data = imf.thin_edges(gradientjk, grad_lengths)
    thin_image = imf.PGMFile(grad_max, thin_data)
    low_threshold = 0.1*thin_image.max_shade
    high_threshold = 0.18*thin_image.max_shade
    suppr_data = imf.suppress_noise(thin_image.data, low_threshold, high_threshold)
    suppr_image = imf.PGMFile(thin_image.max_shade, suppr_data)
    return suppr_image

def readFaces():
    file_names = os.listdir(PATH_TO_FACES)
    d = np.zeros((SIZE**2, len(file_names)), dtype=np.int32)
    print(f"\tDetected {len(file_names)} files")

    for i, filename in enumerate(file_names, start=0):
        img = imf.read_image(PATH_TO_FACES + filename)
        print(f"\tDetecting edges: {i} of {len(file_names)} ")
        img = detectEdges(img)
        max_shade, data = img
        reshaped = data.reshape(-1)
        d[:,i] = reshaped

    
    return d, file_names

def setData():
    global eigenfaces
    global weights
    global mean
    global file_names

    print("\tReading faces...")
    d, file_names = readFaces()
    n = d.shape[1]
    
    print("\tBuilding library...")

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

    print("\tdone.\n\tSaving data...")
    if not os.path.exists(PATH_TO_DATA): os.mkdir(PATH_TO_DATA)
    np.save(PATH_TO_DATA + EIGEN_FACES + '.npy', eigenfaces)
    np.save(PATH_TO_DATA + WEIGHTS + '.npy', weights)
    np.save(PATH_TO_DATA + MEAN + '.npy', mean)
    np.save(PATH_TO_DATA + FILE_NAMES + '.npy', file_names)
    

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
def testImage(file_name_to_test):
    global eigenfaces
    global weights
    global mean
    global file_names

    img = imf.read_image(file_name_to_test)
    max_shade, data = img
    data = data.flatten()

    z = data - mean # subtract mean from image data

    w = findWeight(eigenfaces, z)

    distances = [0] * len(weights) # the distance vector

    for i in range(len(weights)):
        distances[i] = la.norm(weights[i] - w)

    d = np.amin(distances) # the minimal distance to a pic from library
    index = np.where(distances == d)[0][0]

    return d, index


def detectFace(filename):
    print("\n----------------")
    print("Face Or Not Face")
    print("----------------\n")
    print(f"Testing image {filename}\n")
    try:
        print("Loading data...")
        getData()
    except:
        print("No data")
        print("Setting library...")
        setData()
    finally:
        # Test an image
        print("Testing...\n")
        d, ind = testImage(filename)

        print("The closest image is filename=" + file_names[ind])
        print(f"The distance is d={d}\n")

file_to_test = "./Faces/face_1.pgm"

if len(sys.argv) > 1:
    file_to_test = sys.argv[1]

detectFace(file_to_test)